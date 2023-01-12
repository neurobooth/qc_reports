"""
Generate PDF reports of device timing information.
"""
import matplotlib
import matplotlib.pyplot as plt
import os
from collections import defaultdict

import pandas as pd
from typing import Dict, List
from tqdm import tqdm
from datetime import datetime

from neurobooth_analysis_tools.data.files import default_source_directories, discover_session_directories
from neurobooth_analysis_tools.data.files import FileMetadata, parse_files, parse_session_id, discover_associated_files
from neurobooth_analysis_tools.data.types import NeuroboothDevice, DataException
import neurobooth_analysis_tools.data.hdf5 as hdf5
from neurobooth_analysis_tools.data.json import parse_iphone_json
from neurobooth_analysis_tools.io import make_directory

from neurobooth_reports.settings import ReportSettings
from neurobooth_reports.pdf import TaskReport
from neurobooth_reports.plots.units import figsize_mm
from neurobooth_reports.plots.timing import time_offset_plot, time_stability_plot


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


def main() -> None:
    settings = ReportSettings()
    sessions, session_dirs = discover_session_directories(default_source_directories())
    # TODO: Add potential filter for subject/date here

    matplotlib.rcParams.update({'font.size': 8})
    for s, sdir in tqdm(zip(sessions, session_dirs), desc="Generating timing reports", unit='sessions'):
        generate_session_reports(s, sdir, settings)
        return  # TODO: Remove


def generate_session_reports(session: str, session_dir: str, settings: ReportSettings) -> None:
    file_metadata = parse_files(session_dir)
    file_metadata = filter(lambda f: f.extension == '.hdf5', file_metadata)
    file_metadata = sorted(file_metadata, key=lambda f: (f.task.value, DEVICE_ORDER[f.device], f.device_info))
    file_metadata = list(file_metadata)  # Materialize iterator
    tasks = set(m.task for m in file_metadata)

    report_dir = os.path.join(settings.report_dir, session)
    report_paths = {task: os.path.join(report_dir, f'timing_report_{task.name}.pdf') for task in tasks}
    if not should_run_reports(session, list(report_paths.values())):
        return

    make_directory(report_dir)
    for task, report_path in tqdm(report_paths.items(), desc='Creating Report', unit='tasks', leave=False):
        task_files = list(filter(lambda f: f.task == task, file_metadata))

        pdf = TaskReport(session, task)
        for file in task_files:
            create_device_page(pdf, file)
        pdf.output(report_path)


def should_run_reports(session: str, report_paths: List[str]) -> bool:
    _, session_date = parse_session_id(session)
    if (datetime.now() - session_date).days <= RERUN_DAYS:
        return True  # Always re-run reports for recent sessions to catch missing data

    missing_report = not all([os.path.exists(path) for path in report_paths])
    return missing_report  # Run if any report is missing for the session


def create_device_page(pdf: TaskReport, file: FileMetadata) -> None:
    device = file.device
    if device == NeuroboothDevice.Yeti:
        device_page_yeti(pdf, file)
    elif device == NeuroboothDevice.IPhone:
        device_page_iphone(pdf, file)
    else:
        device_page_unimplemented(pdf, file)


def device_page_yeti(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    data = hdf5.extract_yeti(hdf5.load_neurobooth_file(file), include_event_flags=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize_mm(pdf.epw, pdf.epw/3))
    time_stability_plot(ax, data['Time_LSL'])
    ax.set_title('Interpolated LSL Time')
    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)


def device_page_iphone(pdf: TaskReport, file: FileMetadata) -> None:
    data = hdf5.extract_iphone(hdf5.load_neurobooth_file(file), include_event_flags=False)

    json_file = discover_associated_files(file, extensions=['.json'])
    if len(json_file) > 1:
        raise DataException(f"Multiple JSON files detected for {file.file_name}.")
    elif len(json_file) == 1:
        json_data = parse_iphone_json(json_file[0])
        data = pd.merge(data, json_data, how='left', on='FrameNum')

    ############################################################
    # Timing Variability
    ############################################################
    pdf.add_device_page(file.device, file.device_info)
    comparisons = [('Time_JSON', 'Time_iPhone'), ('Time_iPhone', 'Time_Unix'), ('Time_Unix', 'Time_LSL')]
    fig_width, fig_height = pdf.epw, pdf.epw / 3 * len(comparisons)
    fig, axs = plt.subplots(len(comparisons), 1, figsize=figsize_mm(fig_width, fig_height))

    for ax, (t1, t2) in zip(axs, comparisons):
        if t1 not in data.columns or t2 not in data.columns:
            continue
        time_offset_plot(ax, data[t1], data[t2], t1, t2)
    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)

    ############################################################
    # Timing Stability
    ############################################################
    pdf.add_device_page(file.device, file.device_info)
    columns = ['Time_JSON', 'Time_iPhone', 'Time_Unix', 'Time_LSL']
    fig_width, fig_height = pdf.epw, pdf.epw / 3.5 * len(columns)

    fig, axs = plt.subplots(len(columns), 1, figsize=figsize_mm(fig_width, fig_height))
    for ax, col in zip(axs, columns):
        if col not in data.columns:
            continue
        time_stability_plot(ax, data[col])
        ax.set_title(col)
    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)


def device_page_unimplemented(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    pdf.cell(w=pdf.epw, txt='Timing reports for this device are not yet implemented.', align='C')


if __name__ == '__main__':
    main()
