"""
Parse package-wide settings and secrets
"""

import os
import json
from typing import Optional, Dict
from importlib import resources
import neurobooth_reports
from neurobooth_analysis_tools.data.database import DatabaseConnectionInfo


class ReportSettings:
    def __init__(self, config_file: Optional[str] = None):
        config = ReportSettings.load_config(config_file)

        self.report_dir = os.path.abspath(config['report_dir'])
        self.summary_dir = os.path.join(self.report_dir, '__SUMMARY__')

        with open(os.path.abspath(config['secrets']), 'r') as f:
            secrets = json.load(f)
        self.database_connection_info = DatabaseConnectionInfo(**secrets['database'])

    @staticmethod
    def load_config(config_file: Optional[str] = None) -> Dict:
        if config_file is None:
            return json.loads(resources.read_text(neurobooth_reports, 'config.json'))
        else:
            with open(config_file, 'r') as f:
                return json.load(f)
