"""
Generate a list of data files for a set of deviations and potential issues.
"""

import os
from tqdm.contrib.concurrent import process_map
from itertools import chain
from enum import IntEnum
from typing import NamedTuple, List, Optional

import neurobooth_analysis_tools.data.files as nb_file
import neurobooth_analysis_tools.data.hdf5 as nb_hdf
from neurobooth_analysis_tools.data.types import DataException

from neurobooth_reports.settings import ReportSettings

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
        with open(self.path, 'w') as f:
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
    logger = IssueLogger(settings)

    _, session_dirs = nb_file.discover_session_directories(nb_file.default_source_directories())
    file_metadata = [nb_file.parse_files(sdir) for sdir in session_dirs]
    file_metadata = list(chain(*file_metadata))  # Flatten list

    check_task_instr_markers(file_metadata, logger)


def check_task_instr_markers(files: List[nb_file.FileMetadata], logger: IssueLogger) -> None:
    files = list(filter(lambda f: f.extension == '.hdf5', files))
    issues = process_map(
        process_check_task_instr_marker,
        files,
        desc='Checking Task/Instr Markers',
        unit='files',
        chunksize=BATCH_SIZE,
    )
    logger.write_issues(issues)


def process_check_task_instr_marker(
        file: nb_file.FileMetadata,
        test_id: str = 'check_task_instr_marker',
) -> Optional[Issue]:
    """Ensure that an HDF5 file includes markers for task and instruction beginning and end."""
    dev = nb_hdf.load_neurobooth_file(file)
    try:
        nb_hdf.extract_task_boundaries(dev)
    except DataException as e:
        return Issue(
            test_id=test_id,
            issue_desc=f'TASK: {e.args[0]}',
            file=file,
        )

    try:
        nb_hdf.extract_instruction_boundaries(dev)
    except DataException as e:
        return Issue(
            test_id=test_id,
            issue_desc=f'INSTR: {e.args[0]}',
            file=file,
        )

    return None


if __name__ == '__main__':
    main()
