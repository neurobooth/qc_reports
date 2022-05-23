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
