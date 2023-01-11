"""
Generate PDF reports of device timing information.
"""

import os
from collections import defaultdict
from typing import Dict
from tqdm import tqdm
from datetime import datetime

from neurobooth_analysis_tools.data.files import default_source_directories, discover_session_directories
from neurobooth_analysis_tools.data.files import parse_files, parse_session_id
from neurobooth_analysis_tools.data.types import NeuroboothDevice, NeuroboothTask
from neurobooth_analysis_tools.io import make_directory

from neurobooth_reports.settings import ReportSettings
from neurobooth_reports.pdf import TaskReport


# Define the order of devices in the report.
# Any devices not defined will be at the end of the report in an arbitrary order (based on the device enum).
DEVICE_ORDER: Dict[NeuroboothDevice, int] = defaultdict(lambda d: d.value + 1e4, {
    NeuroboothDevice.EyeLink: 0,
    NeuroboothDevice.FLIR: 1,
    NeuroboothDevice.RealSense: 2,
    NeuroboothDevice.IPhone: 3,
    NeuroboothDevice.Yeti: 4,
    NeuroboothDevice.Mbient: 5,
    NeuroboothDevice.Mouse: 6,
})


# How many days after a session to continue re-generating reports (to account for partial uploads)
RERUN_DAYS = 3


def main():
    settings = ReportSettings()
    sessions, session_dirs = discover_session_directories(default_source_directories())
    # TODO: Add potential filter for subject/date here
    for s, sdir in tqdm(zip(sessions, session_dirs), desc="Generating timing reports", unit='sessions'):
        generate_session_reports(s, sdir, settings)
        return  # TODO: Remove


def generate_session_reports(session: str, session_dir: str, settings: ReportSettings):
    report_dir = os.path.join(settings.report_dir, session)
    make_directory(report_dir)
    _, session_date = parse_session_id(session)

    file_metadata = parse_files(session_dir)
    file_metadata = filter(lambda f: f.extension == '.hdf5', file_metadata)
    file_metadata = sorted(file_metadata, key=lambda f: (f.task.value, DEVICE_ORDER[f.device], f.device_info))
    file_metadata = list(file_metadata)  # Materialize iterator

    for task in NeuroboothTask:
        task_files = list(filter(lambda f: f.task == task, file_metadata))
        if len(task_files) == 0:
            continue

        report_path = os.path.join(report_dir, f'timing_report_{task.name}.pdf')
        if os.path.exists(report_path) and (datetime.now() - session_date).days > RERUN_DAYS:
            continue  # Do not regenerate existing reports unless they are recent

        pdf = TaskReport(session, task)
        for file in task_files:
            pdf.add_device_page(file.device, file.device_info)
            # TODO: Add device plots to PDF. Try to fit each device to one page.
        pdf.output(report_path)


if __name__ == '__main__':
    main()
