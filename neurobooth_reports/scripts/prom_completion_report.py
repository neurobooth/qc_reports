"""
Generate a list of patient report completion, alongside completion statistics.
"""
import argparse
import datetime
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Connection
from typing import NamedTuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

from neurobooth_reports.settings import ReportSettings
from neurobooth_reports.output import dataframe_to_csv, save_fig


PROM_RC_FORMS: List[str] = [
    'contact',
    'demographic',
    'prom_ataxia_short_form',
    'prom_ataxia',
    'dysarthria_impact_scale',
    'neurobooth_falls',
    'neuro_qol_ue_short_form',
    'neuro_qol_le_short_form',
    'neuro_qol_cognitive_function_short_form',
    'neuro_qol_participate_social_roles_short_form',
    # 'neuro_qol_satisfaction_social_roles_short_form',  # Table is empty
    # 'neuro_qol_stigma_short_form',  # Table is empty
    'neuro_qol_anxiety_short_form',
    'neuro_qol_depression_short_form',
    'neuro_qol_emotional_dyscontrol_short_form',
    'neuro_qol_positive_affect_and_wellbeing_short_form',
    'neuro_qol_fatigue_short_form',
    'neuro_qol_sleep_disturbance_short_form',
    'communicative_participation_item_bank',
    'handedness_questionnaire',
    'chief_short_form',
    'neurobooth_vision_prom_ataxia',
    'promis_10',
    'system_usability_scale',
    'study_feedback',
]

COMPLETION_METRIC_EXCLUDE = [
    'prom_ataxia_short_form',
]


class PromTableSpec(NamedTuple):
    form_name: str
    table_name: str
    start_time_column: str
    end_time_column: str
    completion_column: str
    offset_column: str


def form_name_to_table_spec(form_name: str) -> PromTableSpec:
    """Take advantage of naming regularities to get Neurobooth table and column names from the Redcap form name."""
    return PromTableSpec(
        form_name=form_name,
        table_name=f'rc_{form_name}',
        start_time_column=f'start_time_{form_name}',
        end_time_column=f'end_time_{form_name}',
        completion_column=f'{form_name}_complete',
        offset_column=f'{form_name}_days_offset',
    )


def download_prom_table(spec: PromTableSpec, connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        spec.table_name,
        connection,
        columns=[
            'subject_id',
            'redcap_event_name',
            spec.start_time_column,
            spec.end_time_column,
            spec.completion_column
        ],
    ).convert_dtypes()

    # Convert completion column to Boolean where 2 is yes
    table[spec.completion_column] = (
        ~table[spec.completion_column].isna() &
        (table[spec.completion_column] == 2)
    )

    # Alter event names to fix visit-level joins
    table['redcap_event_name'] = table['redcap_event_name'].str.replace('enrollment', 'v1', regex=False)
    # Some test subjects have weird duplicate entries... Remove these
    table = table.sort_values(spec.end_time_column, na_position='first') \
        .drop_duplicates(['subject_id', 'redcap_event_name'], keep='last') \
        .reset_index(drop=True)

    return table


def download_subject_table(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'subject',
        connection,
        columns=[
            'subject_id',
            'first_name_birth',
            'middle_name_birth',
            'last_name_birth',
            'date_of_birth_subject',
            'gender_at_birth'
        ],
    ).convert_dtypes()

    # Fix gender encoding
    gender_int = table['gender_at_birth'].astype(float).round().astype(int).to_numpy().copy()
    table['gender_at_birth'] = 'F'
    table.loc[gender_int == 1, 'gender_at_birth'] = 'M'
    table['gender_at_birth'] = table['gender_at_birth'].astype('string')

    # Clean up representation of missing names
    for col in ['first_name_birth', 'middle_name_birth', 'last_name_birth',]:
        mask = table[col].str.lower().isin(['none', '-'])
        table.loc[mask, col] = pd.NA

    return table


def download_contact_info(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_contact',
        connection,
        columns=[
            'subject_id',
            'email_contact',
            'phone_number_contact',
            'phone_type_contact',
            'texting_permission_boolean_contact',
            'comments_contact',
            'end_time_contact',
            'contact_complete',
        ],
    ).convert_dtypes()

    # Decode phone type
    contact_int = table['phone_type_contact'].astype('Int64').copy()
    table['phone_type_contact'] = '[Unknown]'
    table.loc[contact_int == 1, 'phone_type_contact'] = 'Mobile'
    table.loc[contact_int == 2, 'phone_type_contact'] = 'Landline'
    table['phone_type_contact'] = table['phone_type_contact'].astype('string')

    # Only keep the most recent contact information for each person
    table = table.loc[table['contact_complete'] == 2]
    table = table.sort_values('end_time_contact', na_position='first')\
        .drop_duplicates('subject_id', keep='last')\
        .reset_index(drop=True)\
        .drop(columns=['contact_complete', 'end_time_contact'])

    return table


def download_visit_dates(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_visit_dates',
        connection,
        columns=[
            'subject_id',
            'redcap_event_name',
            'neurobooth_visit_dates',
        ],
    ).convert_dtypes()

    # Alter event names to fix visit-level joins
    table['redcap_event_name'] = table['redcap_event_name'].str.replace('enrollment', 'v1', regex=False)

    return table


def download_consent_info(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_baseline_data',
        connection,
        columns=[
            'subject_id',
            'test_subject_boolean',
            'subject_mrn',
            'unsecured_email_agreement_boolean',
        ],
    ).convert_dtypes()

    return table


def download_diagnoses(connection: Connection) -> pd.DataFrame:
    # Grab the coding scheme for the diagnosis column
    diagnosis_map = pd.read_sql_query(
        """
        SELECT response_array FROM rc_data_dictionary
        WHERE field_name = 'primary_diagnosis' AND database_table_name = 'clinical'
        """,
        connection
    ).convert_dtypes().to_numpy()[0, 0]
    diagnosis_map = {int(k): v for k, v in diagnosis_map.items()}

    table = pd.read_sql_table(
        'rc_clinical',
        connection,
        columns=[
            'subject_id',
            'end_time_clinical',
            'primary_diagnosis',
        ],
    ).convert_dtypes()

    # Drop rows with missing info
    table = table.dropna(subset='primary_diagnosis')

    # Convert the list of diagnosis keys into a comma-separated string of diagnoses
    table['primary_diagnosis'] = table['primary_diagnosis'].map(
        lambda diags: ', '.join([diagnosis_map[d] for d in sorted(diags)]),
        na_action='ignore'
    ).astype('string')

    # Only keep the most recent diagnosis for each person
    table = table.sort_values('end_time_clinical', na_position='first')\
        .drop_duplicates('subject_id', keep='last')\
        .reset_index(drop=True)\
        .drop(columns=['end_time_clinical'])

    return table


def download_end_of_visit_details(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_end_of_visit_details',
        connection,
        columns=[
            'subject_id',
            'redcap_event_name',
            'visit_complete_boolean',
            'follow_up_interest_boolean',
            'prom_interest_boolean',
            'visit_notes',
        ],
    ).convert_dtypes()

    # Alter event names to fix visit-level joins
    table['redcap_event_name'] = table['redcap_event_name'].str.replace('enrollment', 'v1', regex=False)

    return table


def reorder_column(df: pd.DataFrame, col_name: str, idx: int) -> pd.DataFrame:
    """Rearrange the Data Frame columns so that the specified name is in the specified position."""
    cols = df.columns.to_list()
    cols.insert(idx, cols.pop(cols.index(col_name)))
    return df.loc[:, cols]


def main():
    parse_arguments()
    settings = ReportSettings()

    table_specs = list(map(form_name_to_table_spec, PROM_RC_FORMS))

    # Download data
    engine = create_engine(settings.database_connection_info.postgresql_url())
    with engine.connect() as connection:
        prom_data: Dict[PromTableSpec, pd.DataFrame] = OrderedDict()
        for spec in table_specs:
            prom_data[spec] = download_prom_table(spec, connection)
        subject_data = download_subject_table(connection)
        contact_data = download_contact_info(connection)
        visit_data = download_visit_dates(connection)
        consent_data = download_consent_info(connection)
        diagnosis_data = download_diagnoses(connection)
        eov_data = download_end_of_visit_details(connection)

    # Combine, subject, latest diagnosis, latest contact, consent, and visit info
    subject_data = pd.merge(subject_data, consent_data, how='left', on='subject_id', validate='1:1')
    subject_data = subject_data.loc[~subject_data['test_subject_boolean']]  # Exclude test subjects
    subject_data = subject_data.drop(columns='test_subject_boolean')
    subject_data = pd.merge(subject_data, diagnosis_data, how='left', on='subject_id', validate='1:1')
    subject_data = pd.merge(subject_data, contact_data, how='left', on='subject_id', validate='1:1')
    visit_data = pd.merge(subject_data, visit_data, how='left', on='subject_id', validate='1:m')
    visit_data = visit_data.dropna(subset='neurobooth_visit_dates')  # Exclude rows with no visit date
    visit_data = pd.merge(visit_data, eov_data, how='left', on=['subject_id', 'redcap_event_name'], validate='1:1')

    # Exclude incomplete visits in the past
    visit_data['Future'] = visit_data['neurobooth_visit_dates'] >= datetime.datetime.today()
    visit_data = visit_data.loc[visit_data['visit_complete_boolean'] | visit_data['Future']]
    visit_data = visit_data.drop(columns='visit_complete_boolean')

    # Join PROMs with visit-level data to yield wide table with all relevant info
    visit_prom_data = visit_data.copy()
    for spec, prom in prom_data.items():
        visit_prom_data = pd.merge(
            visit_prom_data, prom,
            how='left', on=['subject_id', 'redcap_event_name'], validate='1:1'
        )
        visit_prom_data[spec.offset_column] = \
            visit_prom_data[spec.end_time_column] - visit_prom_data['neurobooth_visit_dates']
        visit_prom_data[spec.offset_column] /= np.timedelta64(1, 'D')  # Convert to days

    for spec in table_specs:
        # Mark NaN values in PROM completion columns (missing from joins) as uncompleted
        mask = visit_prom_data[spec.completion_column].isna()
        visit_prom_data.loc[mask, spec.completion_column] = False

    # Create reports
    prom_completion_report(visit_data, visit_prom_data, table_specs, settings)
    prom_completion_time_report(visit_prom_data, table_specs, settings)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate device timing reports.")
    return parser.parse_args()


def prom_completion_report(
        visit_data: pd.DataFrame,
        visit_prom_data: pd.DataFrame,
        table_specs: List[PromTableSpec],
        settings: ReportSettings
) -> None:
    """Create a list of survey completion percentages for each visit."""
    completion_report_name = 'prom_completion.csv'
    contact_report_name = 'prom_contact.csv'

    # Arm 1 does not do the PROM Ataxia or DIS. Quick hack: mark them as completed for Arm 1.
    visit_prom_data = visit_prom_data.copy().sort_values(['subject_id', 'neurobooth_visit_dates'])
    arm1_mask = visit_prom_data['redcap_event_name'].str.contains('_arm_1', regex=False)
    visit_prom_data.loc[arm1_mask, 'prom_ataxia_complete'] = True
    visit_prom_data.loc[arm1_mask, 'dysarthria_impact_scale_complete'] = True

    report = visit_prom_data[
        ['subject_id', 'neurobooth_visit_dates', 'redcap_event_name', 'primary_diagnosis', 'Future']
    ].copy()

    # Calculate completion data
    completion_cols = [
        spec.completion_column for spec in table_specs
        if spec.form_name not in COMPLETION_METRIC_EXCLUDE
    ]
    completion_matrix = visit_prom_data[completion_cols].to_numpy(dtype=bool)
    report['PROM_Completion_Pct'] = (completion_matrix.mean(axis=1).round(3) * 100).astype(float)
    report['PROM_Completion_Pct'] = report['PROM_Completion_Pct'].astype(float)
    report['PROM_Completion_Pct'] = report['PROM_Completion_Pct'].apply(lambda x: f"{x:.4f}")
    report['PROM_All_Complete'] = completion_matrix.all(axis=1)

    # Detect longitudinal visits
    report['Visit_Num'] = report.groupby('subject_id')['neurobooth_visit_dates'].rank(ascending=True, method='max')

    for spec in table_specs:
        report[spec.completion_column] = visit_prom_data[spec.completion_column]
        report[spec.start_time_column] = visit_prom_data[spec.start_time_column]
        report[spec.end_time_column] = visit_prom_data[spec.end_time_column]
        report[spec.offset_column] = visit_prom_data[spec.offset_column]

    dataframe_to_csv(
        os.path.join(settings.summary_dir, completion_report_name),
        report.sort_values(['subject_id', 'neurobooth_visit_dates'])
    )

    # Augment table to show upcoming appointments, then limit to past sessions
    future_visits = report.loc[
        report['Future'], ['subject_id', 'neurobooth_visit_dates']
    ].set_index('subject_id').to_dict()['neurobooth_visit_dates']
    report = report.loc[~report['Future']]
    report = report.drop(columns='Future')
    report['Upcoming_Visit_Date'] = report['subject_id'].map(future_visits)

    # Add contact info
    report = report.drop(columns='redcap_event_name')  # Will be duplicated if not dropped from one table
    report = report.drop(columns='primary_diagnosis')  # Will be duplicated if not dropped from one table
    report = pd.merge(visit_data, report, on=['subject_id', 'neurobooth_visit_dates'], validate='1:1')
    report = report.drop(columns='Future')
    report = reorder_column(report, 'neurobooth_visit_dates', 1)
    report = reorder_column(report, 'Upcoming_Visit_Date', 2)
    report = reorder_column(report, 'Visit_Num', 3)
    report = reorder_column(report, 'redcap_event_name', 4)
    report = reorder_column(report, 'primary_diagnosis', 5)

    # Only report the latest session if proms uncompleted
    report: pd.DataFrame = report.sort_values(['Upcoming_Visit_Date', 'neurobooth_visit_dates'], ascending=False)\
        .drop_duplicates('subject_id', keep='first')\
        .reset_index(drop=True)
    report = report.loc[~report['PROM_All_Complete']]

    dataframe_to_csv(os.path.join(settings.summary_dir, contact_report_name), report)


def prom_completion_time_report(
        visit_prom_data: pd.DataFrame,
        table_specs: List[PromTableSpec],
        settings: ReportSettings,
) -> None:
    """Create box plots + statistics for how long each survey takes."""
    fig_path = os.path.join(settings.summary_dir, 'prom_completion_time.png')
    csv_path = os.path.join(settings.summary_dir, 'prom_completion_time_stats.csv')

    form = []
    subject_id = []
    is_control = []
    completion_times = []
    for spec in table_specs:
        time_delta = visit_prom_data[spec.end_time_column] - visit_prom_data[spec.start_time_column]

        # Filter deltas to just what it makes sense to analyze
        mask = ~time_delta.isna()
        mask &= visit_prom_data[spec.completion_column]
        mask &= ~visit_prom_data.duplicated(['subject_id', spec.start_time_column, spec.end_time_column])
        time_delta = time_delta.loc[mask]

        # Convert to minutes
        time_delta /= np.timedelta64(1, 'm')
        time_delta = time_delta.to_numpy('float64')

        form.extend(np.full(time_delta.shape, spec.form_name))
        subject_id.extend(visit_prom_data.loc[mask, 'subject_id'])
        is_control.extend(visit_prom_data.loc[mask, 'primary_diagnosis'].str.lower() == 'control')
        completion_times.extend(time_delta)

    completion_time_df = pd.DataFrame.from_dict({
        'Form': form,
        'Subject ID': subject_id,
        'Control': is_control,
        'Completion Time': completion_times,
    })
    long_times = completion_time_df['Completion Time'] > 60
    zero_times = completion_time_df['Completion Time'] == 0
    completion_time_df_long = completion_time_df.loc[long_times].copy()
    completion_time_df = completion_time_df.loc[~(long_times | zero_times)]

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    sns.boxplot(completion_time_df, x='Completion Time', y='Form', hue='Control', orient='h', ax=ax)
    ax.set_xlim([0, 20])
    ax.set_xticks(np.linspace(0, 20, 5))
    ax.set_xlabel('Completion Time (min)')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    fig.tight_layout()
    save_fig(fig_path, fig, close=True)

    groupby = ['Form', 'Control']
    stats_df = completion_time_df.groupby(groupby).agg(
        N=('Completion Time', 'size'),
        mean=('Completion Time', 'mean'),
        std=('Completion Time', 'std'),
        pct_10=('Completion Time', lambda x: np.percentile(x, 10)),
        pct_25=('Completion Time', lambda x: np.percentile(x, 25)),
        pct_50=('Completion Time', lambda x: np.percentile(x, 50)),
        pct_75=('Completion Time', lambda x: np.percentile(x, 75)),
        pct_90=('Completion Time', lambda x: np.percentile(x, 90)),
    )
    long_stats_df = completion_time_df_long.groupby(groupby).agg(
        n_gt_hr=('Completion Time', 'size'),
        n_gt_24hr=('Completion Time', lambda x: np.sum(x > (60 * 24))),
        n_gt_week=('Completion Time', lambda x: np.sum(x > (60 * 24 * 7))),
        n_gt_30day=('Completion Time', lambda x: np.sum(x > (60 * 24 * 30))),
    )
    # Join will use the index information to perform the join
    stats_df = stats_df.join(long_stats_df, how='left', validate='1:1')
    stats_df = stats_df.fillna(0)

    stats_df = stats_df.reset_index()
    dataframe_to_csv(csv_path, stats_df)


if __name__ == '__main__':
    main()
