"""
Parse package-wide settings and secrets
"""

import os
import json
from typing import NamedTuple, Optional, Dict, List
from importlib import resources
import neurobooth_reports
from neurobooth_analysis_tools.data.database import DatabaseConnectionInfo


class EmailSettings(NamedTuple):
    subject_prefix: str
    from_addr: str
    to_addr: List[str]


class ReportSettings:
    def __init__(self, config_file: Optional[str] = None):
        config = ReportSettings.load_config(config_file)

        self.report_dir = os.path.abspath(config['report_dir'])
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

        self.summary_dir = os.path.join(self.report_dir, '__SUMMARY__')
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        with open(os.path.abspath(config['secrets']), 'r') as f:
            secrets = json.load(f)
        self.database_connection_info = DatabaseConnectionInfo(**secrets['database'])

        self.email = self._load_email_settings(os.path.abspath(config['mailing_list']))

    @staticmethod
    def load_config(config_file: Optional[str] = None) -> Dict:
        if config_file is None:
            return json.loads(resources.read_text(neurobooth_reports, 'config.json'))
        else:
            with open(config_file, 'r') as f:
                return json.load(f)

    def _load_email_settings(self, mailing_list_path: str) -> EmailSettings:
        # Create an empty file if it does not currently exist
        if not os.path.exists(mailing_list_path):
            with open(mailing_list_path, 'w') as f:
                json.dump({
                    "Subject Prefix": "[Neurobooth] ",
                    "From": "",
                    "To": [""],
                }, f)

        with open(mailing_list_path, 'r') as f:
            mailing_list = json.load(f)

        return EmailSettings(
            subject_prefix=mailing_list['Subject Prefix'],
            from_addr=mailing_list['From'],
            to_addr=mailing_list['To']
        )
