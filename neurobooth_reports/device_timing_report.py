"""
Generate PDF reports of device timing information.
"""

import os
from neurobooth_analysis_tools.data.files import default_source_directories, discover_session_directories
from neurobooth_analysis_tools.data.files import parse_files


def main():
    return
    # _, session_dirs = discover_session_directories(default_source_directories())
    # for sdir in session_dirs:
    #     generate_session_reports(sdir)


def generate_session_reports(session_dir: str):
    file_metadata = parse_files(session_dir)


if __name__ == '__main__':
    main()
