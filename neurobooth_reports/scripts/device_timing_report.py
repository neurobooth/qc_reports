"""
Generate PDF reports of device timing information.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, List, Tuple
from tqdm.contrib.concurrent import process_map
import datetime
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

from neurobooth_analysis_tools.data.files import default_source_directories, discover_session_directories
from neurobooth_analysis_tools.data.files import FileMetadata, parse_files, parse_session_id, discover_associated_files
from neurobooth_analysis_tools.data.types import NeuroboothDevice, DataException
import neurobooth_analysis_tools.data.hdf5 as hdf5
from neurobooth_analysis_tools.data.json import parse_iphone_json
from neurobooth_analysis_tools.io import make_directory
from neurobooth_analysis_tools.plot.shade import shade_mask

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


def main() -> None:
    args = parse_arguments()
    settings = ReportSettings()
    sessions, session_dirs = discover_session_directories(default_source_directories())
    sessions, session_dirs = np.array(sessions), np.array(session_dirs)

    sort_idx = np.argsort(sessions)
    sessions = sessions[sort_idx]
    session_dirs = session_dirs[sort_idx]

    if args.subject is not None:
        mask = np.array([parse_session_id(s)[0] for s in sessions]) == args.subject
        sessions = sessions[mask]
        session_dirs = session_dirs[mask]

    if args.date is not None:
        mask = np.array([parse_session_id(s)[1] for s in sessions]) == args.date
        sessions = sessions[mask]
        session_dirs = session_dirs[mask]

    if sessions.shape[0] == 0:
        print('No sessions meet filter criteria!')

    matplotlib.rcParams.update({'font.size': 8})
    process_map(
        generate_session_reports,
        [str(s) for s in sessions],
        [str(sdir) for sdir in session_dirs],
        [settings for _ in range(sessions.shape[0])],
        [args.rerun_days for _ in range(sessions.shape[0])],
        desc='Generating timing reports',
        unit='sessions',
        chunksize=1,
        max_workers=args.n_cpu,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate device timing reports.")

    group = parser.add_argument_group(title="General Options")
    group.add_argument(
        '--rerun-days',
        type=int,
        default=3,
        help="Always rerun reports for session dates less than this many days ago."
    )
    group.add_argument(
        '--n-cpu',
        type=int,
        default=4,
        help="How many parallel processes to run when generating reports."
    )

    group = parser.add_argument_group(
        title="Filters",
        description="Only generate reports that match the specified criteria."
    )
    group.add_argument(
        '--subject',
        type=str,
        default=None,
        help="Only generate reports for the given subject."
    )
    group.add_argument(
        '--date',
        type=datetime.date.fromisoformat,
        default=None,
        help="Only generate reports for the given session date."
    )

    return parser.parse_args()


def generate_session_reports(
        session: str,
        session_dir: str,
        settings: ReportSettings,
        rerun_days: int,
) -> None:
    file_metadata = parse_files(session_dir)
    file_metadata = filter(lambda f: f.extension == '.hdf5', file_metadata)
    file_metadata = sorted(file_metadata, key=lambda f: (f.task.value, DEVICE_ORDER[f.device], f.device_info))
    file_metadata = list(file_metadata)  # Materialize iterator
    tasks = set(m.task for m in file_metadata)

    report_dir = os.path.join(settings.report_dir, session)
    report_paths = {task: os.path.join(report_dir, f'timing_report_{task.name}.pdf') for task in tasks}
    if not should_run_reports(session, list(report_paths.values()), rerun_days):
        return

    make_directory(report_dir)
    for task, report_path in report_paths.items():
        task_files = list(filter(lambda f: f.task == task, file_metadata))

        pdf = TaskReport(session, task)
        for file in task_files:
            create_device_page(pdf, file)
        pdf.output_file(report_path)


def should_run_reports(session: str, report_paths: List[str], rerun_days: int) -> bool:
    _, session_date = parse_session_id(session)
    if (datetime.datetime.today() - session_date).days <= rerun_days:
        return True  # Always re-run reports for recent sessions to catch missing data

    missing_report = not all([os.path.exists(path) for path in report_paths])
    return missing_report  # Run if any report is missing for the session


def create_device_page(pdf: TaskReport, file: FileMetadata) -> None:
    device = file.device
    if device == NeuroboothDevice.Yeti:
        device_page_yeti(pdf, file)
    elif device == NeuroboothDevice.IPhone:
        device_page_iphone(pdf, file)
    elif device == NeuroboothDevice.EyeLink:
        device_page_eyelink(pdf, file)
    else:
        device_page_unimplemented(pdf, file)


def extract_data(pdf: TaskReport, file: FileMetadata, extract_func: Callable) -> Tuple[bool, Optional[pd.DataFrame]]:
    try:
        device = hdf5.load_neurobooth_file(file)
        if device.data.time_series.shape[0] == 0 or device.data.time_series.shape[0] == 1:
            pdf.cell(w=pdf.epw, txt=f'ERROR: Insufficient time-series data.', align='C')
            return False, None
        data = extract_func(device, include_event_flags=False)
    except Exception as e:
        pdf.cell(w=pdf.epw, txt=f'ERROR: {e.args[0]}', align='C')
        return False, None

    # Don't try to parse instruction/task period if marker data is missing
    if device.marker.time_series.shape[0] == 0:
        return True, data

    try:
        data['Flag_Instructions'] = hdf5.create_instruction_mask(device, data['Time_LSL'])
    except DataException:
        pass

    try:
        data['Flag_Task'] = hdf5.create_task_mask(device, data['Time_LSL'])
    except DataException:
        pass

    return True, data


def plot_event_shading(ax: plt.Axes, data: pd.DataFrame):
    ts = np.arange(data.shape[0])
    if 'Flag_Instructions' in data.columns:
        plot_kws = {'color': 'g', 'alpha': 0.3}
        shade_mask(ax, data['Flag_Instructions'], ts, plot_kws=plot_kws)
    if 'Flag_Task' in data.columns:
        plot_kws = {'color': 'y', 'alpha': 0.3}
        shade_mask(ax, data['Flag_Task'], ts, plot_kws=plot_kws)


def device_page_yeti(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    success, data = extract_data(pdf, file, hdf5.extract_yeti)
    if not success:
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize_mm(pdf.epw, pdf.epw/3))
    time_stability_plot(ax, data['Time_LSL'])
    plot_event_shading(ax, data)
    ax.set_title('Interpolated LSL Time')
    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)


def device_page_iphone(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    success, data = extract_data(pdf, file, hdf5.extract_iphone)
    if not success:
        return

    json_file = discover_associated_files(file, extensions=['.json'])
    if len(json_file) > 1:
        raise DataException(f"Multiple JSON files detected for {file.file_name}.")
    elif len(json_file) == 1:
        json_data = parse_iphone_json(json_file[0])
        data = pd.merge(data, json_data, how='left', on='FrameNum')

    ############################################################
    # Timing Variability
    ############################################################
    comparisons = [('Time_JSON', 'Time_iPhone'), ('Time_iPhone', 'Time_Unix'), ('Time_Unix', 'Time_LSL')]
    fig_width, fig_height = pdf.epw, pdf.epw / 3 * len(comparisons)
    fig, axs = plt.subplots(len(comparisons), 1, figsize=figsize_mm(fig_width, fig_height))

    for ax, (t1, t2) in zip(axs, comparisons):
        if t1 not in data.columns or t2 not in data.columns:
            continue
        time_offset_plot(ax, data[t1], data[t2], t1, t2)
        plot_event_shading(ax, data)
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
        plot_event_shading(ax, data)
        ax.set_title(col)
    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)


def device_page_eyelink(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    success, data = extract_data(pdf, file, hdf5.extract_eyelink)
    if not success:
        return

    fig_width, fig_height = pdf.epw, pdf.epw / 3 * 3
    fig, axs = plt.subplots(3, 1, figsize=figsize_mm(fig_width, fig_height))

    # Timing Variability
    time_offset_plot(axs[0], data['Time_NUC'], data['Time_LSL'], 'Time_NUC', 'Time_LSL')
    plot_event_shading(axs[0], data)

    # Timing Stability - NUC
    time_stability_plot(axs[1], data['Time_NUC'])
    plot_event_shading(axs[1], data)
    axs[1].set_title('Time_NUC')

    # Timing Stability - LSL
    time_stability_plot(axs[2], data['Time_LSL'])
    plot_event_shading(axs[2], data)
    axs[2].set_title('Time_LSL')

    fig.tight_layout()
    pdf.add_figure(fig, close=True, full_width=True)


def device_page_unimplemented(pdf: TaskReport, file: FileMetadata) -> None:
    pdf.add_device_page(file.device, file.device_info)
    pdf.cell(w=pdf.epw, txt='Timing reports for this device are not yet implemented.', align='C')


if __name__ == '__main__':
    main()
