import os
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from typing import NamedTuple, List
from tqdm.contrib.concurrent import process_map
from itertools import chain

from neurobooth_analysis_tools.data.files import (
    default_source_directories, discover_session_directories, parse_files, FileMetadata
)
from neurobooth_analysis_tools.data.types import NeuroboothDevice, NeuroboothTask
from neurobooth_analysis_tools.data.database import DatabaseConnection
from neurobooth_analysis_tools.data.hdf5 import load_neurobooth_file, extract_mbient

from neurobooth_reports.settings import ReportSettings


def main():
    # Script initialization
    args = parse_arguments()
    settings = ReportSettings()
    if args.output is None:
        args.output = os.path.join(settings.summary_dir, 'mbient_timing_matrix.csv')
    else:
        args.output = os.path.abspath(args.output)

    # Discover and attempt to compute basic sampling rate stats for every Mbient file
    mbient_files = discover_mbient_files(args, settings)
    results = process_map(
        process_mbient_file,
        mbient_files,
        desc='Extracting Data',
        unit='files',
        chunksize=1,
        max_workers=args.n_cpu,
    )

    # Display processing error statistics
    errors = ~np.array([r.success for r in results], dtype=bool)
    print(f"{errors.mean() * 100:.1f}% (N={errors.sum():d}) of Mbient files could not be processed.")

    # Arrange results into a data frame for analysis and export to file
    result_df = organize_results(results)
    result_df.to_csv(args.output, index=True)
    print(f"Done. Output saved to {args.output}")


def discover_mbient_files(args: argparse.Namespace, settings: ReportSettings) -> List[FileMetadata]:
    # Discover and process all neurobooth filenames
    _, session_dirs = discover_session_directories(default_source_directories())
    file_metadata = list(chain(*process_map(
        parse_files,
        session_dirs,
        desc='Parsing File Names',
        unit='sessions',
        chunksize=1,
        max_workers=args.n_cpu,
    )))

    # Filter out test subjects and non-mbient files
    test_subjects = DatabaseConnection(settings.database_connection_info).get_test_subjects()
    file_metadata = filter(lambda m: int(m.subject_id) >= 100100, file_metadata)
    file_metadata = filter(lambda m: m.subject_id not in test_subjects, file_metadata)
    file_metadata = filter(lambda m: m.extension == '.hdf5', file_metadata)
    file_metadata = filter(lambda m: m.device == NeuroboothDevice.Mbient, file_metadata)
    return list(file_metadata)  # Resolve filters


class ProcessResult(NamedTuple):
    subject_id: str
    task: NeuroboothTask
    task_time: datetime
    device_location: str
    success: bool
    mean_sample_delta: float
    std_sample_delta: float
    min_sample_delta: float
    max_sample_delta: float


def process_mbient_file(file: FileMetadata) -> ProcessResult:
    success = False
    try:
        data = load_neurobooth_file(file)
        data = extract_mbient(data)
        data = data.loc[data['Flag_Task']]  # Only include task data in statistics

        ts = data['Time_Mbient'].to_numpy()
        if ts.shape[0] < 30:
            raise Exception(f'Only f{ts.shape[0]:d} samples in time-series')

        ts_diff = np.diff(ts)
        mean_sample_delta = ts_diff.mean()
        std_sample_delta = ts_diff.std()
        min_sample_delta = ts_diff.min()
        max_sample_delta = ts_diff.max()

        success = True
    except:
        mean_sample_delta = np.nan
        std_sample_delta = np.nan
        min_sample_delta = np.nan
        max_sample_delta = np.nan

    return ProcessResult(
        subject_id=file.subject_id,
        task=file.task,
        task_time=file.datetime,
        device_location=file.device_info,
        success=success,
        mean_sample_delta=mean_sample_delta,
        std_sample_delta=std_sample_delta,
        min_sample_delta=min_sample_delta,
        max_sample_delta=max_sample_delta,
    )


def organize_results(results: List[ProcessResult]) -> pd.DataFrame:
    # Only keep results from successful processes (and transform into dicts for DataFrame constructor)
    results = [r._asdict() for r in results if r.success]

    # Organize the results into a DataFrame and export
    df = pd.DataFrame(results)
    df = df.drop(columns='success')  # No longer needed because of above filter
    df = df.rename(columns={  # Shorten columns names
        'mean_sample_delta': 'mean',
        'std_sample_delta': 'std',
        'min_sample_delta': 'min',
        'max_sample_delta': 'max',
    })
    df = df.pivot(  # Convert the "long-form" table into a "wide-form" table for easier analysis
        index=['subject_id', 'task', 'task_time'],
        columns=['device_location'],
        values=['mean', 'std', 'min', 'max'],
    )
    df = df.swaplevel(0, 1, axis='columns')  # Column index should have location as top level
    df = df.sort_index(axis='columns', level=0, sort_remaining=False)  # Better column organization
    df = df.sort_index(axis='index', level=0)  # Human-readable row organization
    return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a matrix of Mbient sampling rate features.")
    parser.add_argument(
        '-O', '--output',
        default=None,
        type=str,
        help='Override the default output file path.'
    )
    parser.add_argument(
        '-C', '--n-cpu',
        default=4,
        type=int,
        help=f'The number of processes to use for parallel processing. ({os.cpu_count():d} available)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
