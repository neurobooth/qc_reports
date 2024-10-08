# Neurobooth PROM Reports

A set of scripts to generate Neurobooth PROM reports and upload them to dropbox.

To install run the following command:

```pip install -e git+https://github.com/neurobooth/qc_reports.git#egg=neurobooth_reports```

Filesystem and installation idiosyncracies:

* If you get a directory not found error, you might need to edit the shebang line of prom_completion_report executable in conda environment's bin to remove 'local_mount' and start from 'space' instead. The executable should be in ```miniforge3/envs/<env name>/bin/prom_completion_report```
* If you are not actively setting --prefix/--target/--src or other similar flags during pip installation, you might need to be in /space/... folder tree instead of /local_mount/space... folder tree - this is for pip to pick the right posixpath so that the installation is available across servers in the unified filesystem.


# qc_reports

Scripts to generate session level pdf reports for data collected at Neurobooth.
Each pdf contains all data for each session, one page per task, with each page containing subplots for each sensor type.

## Script descriptions

__generate_session_data_report.py__ 

This script generates pdf report for a single session, taking the subject id and session date as input argument. The user is intended to check Neurobooth Explorer to ensure data for the session exists and can be viewed through Neurobooth Explorer.

Run: ```python generate_session_data_report.py -h```

Output:
```
usage: generate_session_data_report.py [-h] --subj_id SUBJ_ID --session_date SESSION_DATE [--data_dir DATA_DIR] [--processed_data_dir PROCESSED_DATA_DIR]
                                       [--save_dir SAVE_DIR]

Generates patient data reports as pdf files
Requires sesssion id in the form of 'subject-id_session-date'
--subj_id and --session_date are required arguments
Searches default location for session data
use -h or --help flag for help

optional arguments:
  -h, --help            show this help message and exit
  --subj_id SUBJ_ID
  --session_date SESSION_DATE
  --data_dir DATA_DIR
  --processed_data_dir PROCESSED_DATA_DIR
  --save_dir SAVE_DIR
```

Required flags:
* --subj_id
* --session_date

Optional flags:
* --data_dir : Default = Neurobooth data directory
* --processed_data_dir : Default = Neurobooth processed data directory
* --save_dir : Default = Neurobooth qc_reports directory

__generate_QC_report.py__

This scripts generates pdf reports for all session data available in the Neurobooth data folder. All arguments are optional. If report exists at save location, the script skips to next session. If processed data exists for session, the pdf report is recreated even if it exists.

Run: ```python generate_QC_report.py -h```

Output:
```
usage: generate_QC_report.py [-h] [--data_dir DATA_DIR] [--processed_data_dir PROCESSED_DATA_DIR] [--save_dir SAVE_DIR]

Generates patient data reports for all sesssion_ids
found in data_dir and saves pdf reports to save_dir
All arguments are optional
use -h or --help flag for help

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR
  --processed_data_dir PROCESSED_DATA_DIR
  --save_dir SAVE_DIR
```

Required flags: None

Optional flags:
* --data_dir : Default = Neurobooth data directory
* --processed_data_dir : Default = Neurobooth processed data directory
* --save_dir : Default = Neurobooth qc_reports directory
