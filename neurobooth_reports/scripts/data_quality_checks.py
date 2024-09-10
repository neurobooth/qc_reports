"""
Generate a list of data files for a set of deviations and potential issues.
"""

import os
from tqdm.contrib.concurrent import process_map
from itertools import chain
from typing import NamedTuple, List, Optional

import neurobooth_analysis_tools.data.files as nb_file
import neurobooth_analysis_tools.data.hdf5 as nb_hdf
from neurobooth_analysis_tools.data.types import DataException

from neurobooth_reports.settings import ReportSettings
from neurobooth_reports.output import get_file_descriptor

# How many tasks each worker process should handle at a time.
# Affects progress bar resolution and maybe performance (larger batches minimize cross-process IO).
BATCH_SIZE = 32


class Issue(NamedTuple):
    test_id: str
    issue_desc: str
    file: nb_file.FileMetadata


class IssueLogger:
    def __init__(self, settings: ReportSettings):
        self.path = os.path.join(settings.summary_dir, 'data_quality_issues.csv')
        with open(get_file_descriptor(self.path), 'w') as f:
            f.write('Test,Issue,Subject ID,Session Date,Task,Device,File Path\n')

    def write_issues(self, issues: List[Optional[Issue]]):
        issues = [i for i in issues if i is not None]
        with open(self.path, 'a') as f:
            for issue in issues:
                f.write(f'"{issue.test_id}",')
                f.write(f'"{issue.issue_desc}",')
                f.write(f'"{issue.file.subject_id}",')
                f.write(f'"{issue.file.datetime.strftime("%Y-%m-%d")}",')
                f.write(f'"{issue.file.task.name}",')
                f.write(f'"{issue.file.device.name}",')
                f.write(f'"{nb_hdf.resolve_filename(issue.file)}"\n')


def main():
    settings = ReportSettings()

    sessions, session_dirs = nb_file.discover_session_directories(nb_file.default_source_directories())
    file_metadata = [nb_file.parse_files(sdir) for sdir in session_dirs]
    file_metadata = chain(*file_metadata)  # Flatten list
    file_metadata = list(sorted(file_metadata, key=lambda m: (m.subject_id, m.datetime, m.task, m.device)))
    hdf_files = list(filter(lambda f: f.extension == '.hdf5', file_metadata))

    issues = process_map(
        run_hdf_checks,
        hdf_files,
        desc='Checking For Errors',
        unit='files',
        chunksize=BATCH_SIZE,
    )
    issues = list(chain(*issues))
    IssueLogger(settings).write_issues(issues)


def run_hdf_checks(file: nb_file.FileMetadata) -> List[Optional[Issue]]:
    """Run all the data-related checks in a way that only loads the files once."""
    try:
        device = nb_hdf.load_neurobooth_file(file)
    except Exception as e:
        return [Issue(
            test_id='file_load',
            issue_desc=f'Unable to load file: {e.args[0]}',
            file=file,
        )]

    try:
        return [
            check_footer(file, device),
            check_task_instr_marker(file, device),
            check_sufficient_data(file, device),
        ]
    except Exception as e:
        return [Issue(
            test_id='unexpected_exception',
            issue_desc=f'ERROR: {e.args[0]}',
            file=file,
        )]


def check_task_instr_marker(
        file: nb_file.FileMetadata,
        device: nb_hdf.Device,
        test_id: str = 'check_task_instr_marker',
) -> Optional[Issue]:
    """Ensure that an HDF5 file includes markers for task and instruction beginning and end."""
    if device.marker.time_series.shape[0] == 0:
        return Issue(
            test_id=test_id,
            issue_desc=f'Marker time-series is empty.',
            file=file,
        )

    try:
        nb_hdf.extract_task_boundaries(device)
    except DataException as e:
        return Issue(
            test_id=test_id,
            issue_desc=f'TASK: {e.args[0]}',
            file=file,
        )

    try:
        nb_hdf.extract_instruction_boundaries(device)
    except DataException as e:
        return Issue(
            test_id=test_id,
            issue_desc=f'INSTR: {e.args[0]}',
            file=file,
        )

    return None


def check_sufficient_data(
        file: nb_file.FileMetadata,
        device: nb_hdf.Device,
        test_id: str = 'check_sufficient_data',
) -> Optional[Issue]:
    N = device.marker.time_series.shape[0]

    if N == 0:
        return Issue(
            test_id=test_id,
            issue_desc=f'Device time-series is empty!',
            file=file,
        )

    if N == 1:
        return Issue(
            test_id=test_id,
            issue_desc=f'Device time-series has single sample!',
            file=file,
        )

    return None


def check_footer(
        file: nb_file.FileMetadata,
        device: nb_hdf.Device,
        test_id: str = 'check_footer',
) -> Optional[Issue]:
    if device.data.footer is None:
        return Issue(
            test_id=test_id,
            issue_desc=f'Device missing footer.',
            file=file,
        )

    if device.marker.footer is None:
        return Issue(
            test_id=test_id,
            issue_desc=f'Marker missing footer.',
            file=file,
        )

    return None


if __name__ == '__main__':
    main()
