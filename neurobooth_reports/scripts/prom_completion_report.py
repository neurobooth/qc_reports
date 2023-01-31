"""
Generate a list of patient report completion, alongside completion statistics.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Connection
from typing import NamedTuple, List, Dict
from neurobooth_reports.settings import ReportSettings


PROM_RC_FORMS: List[str] = [
    'handedness_questionnaire',
    'chief_short_form',
    # 'communicative_participation_item_bank',  # Table is empty
    'dysarthria_impact_scale',
    'neuro_qol_anxiety_short_form',
    'neuro_qol_cognitive_function_short_form',
    'neuro_qol_depression_short_form',
    # 'neuro_qol_emotional_dyscontrol_short_form',  # Table is empty
    'neuro_qol_fatigue_short_form',
    'neuro_qol_le_short_form',
    # 'neuro_qol_participate_social_roles_short_form',  # Table is empty
    'neuro_qol_positive_affect_and_wellbeing_short_form',
    # 'neuro_qol_satisfaction_social_roles_short_form',  # Table is empty
    'neuro_qol_sleep_disturbance_short_form',
    # 'neuro_qol_stigma_short_form',  # Table is empty
    'neuro_qol_ue_short_form',
    'neurobooth_falls',
    'neurobooth_vision_prom_ataxia',
    'prom_ataxia',
    'promis_10',
    'study_feedback',
    'system_usability_scale',
]


class PromTableSpec(NamedTuple):
    form_name: str
    table_name: str
    start_time_column: str
    end_time_column: str
    completion_column: str


def form_name_to_table_spec(form_name: str) -> PromTableSpec:
    """Take advantage of naming regularities to get Neurobooth table and column names from the Redcap form name."""
    return PromTableSpec(
        form_name=form_name,
        table_name=f'rc_{form_name}',
        start_time_column=f'start_time_{form_name}',
        end_time_column=f'end_time_{form_name}',
        completion_column=f'{form_name}_complete'
    )


def download_prom_table(spec: PromTableSpec, connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        spec.table_name,
        connection,
        columns=['subject_id', spec.start_time_column, spec.end_time_column, spec.completion_column],
    ).convert_dtypes()

    # Convert completion column to Boolean where 2 is yes
    table[spec.completion_column] = (
        ~table[spec.completion_column].isna() &
        (table[spec.completion_column] == 2)
    )

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

    return table


def download_contact_table(connection: Connection) -> pd.DataFrame:
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
    table = table.sort_values('end_time_contact').drop_duplicates('subject_id', keep='last').reset_index(drop=True)
    table = table.drop(columns='contact_complete')

    return table


def download_visit_dates(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_visit_dates',
        connection,
        columns=[
            'subject_id',
            'neurobooth_visit_dates',
        ],
    ).convert_dtypes()

    return table


def download_consent_info_table(connection: Connection) -> pd.DataFrame:
    table = pd.read_sql_table(
        'rc_participant_and_consent_information',
        connection,
        columns=[
            'subject_id',
            'test_subject_boolean',
            'subject_mrn',
            'unsecured_email_agreement_boolean',
        ],
    ).convert_dtypes()

    return table


def main():
    settings = ReportSettings()

    table_specs = list(map(form_name_to_table_spec, PROM_RC_FORMS))

    engine = create_engine(settings.database_connection_info.postgresql_url())
    with engine.connect() as connection:
        prom_data: Dict[PromTableSpec, pd.DataFrame] = {
            spec: download_prom_table(spec, connection) for spec in table_specs
        }
        subject_data = download_subject_table(connection)
        contact_data = download_contact_table(connection)
        visit_data = download_visit_dates(connection)
        consent_data = download_consent_info_table(connection)

    subject_data = pd.merge(subject_data, consent_data, how='left', on='subject_id', validate='1:1')
    subject_data = subject_data.loc[~subject_data['test_subject_boolean']]  # Exclude test subjects
    subject_data = pd.merge(subject_data, contact_data, how='left', on='subject_id', validate='1:1')
    visit_data = pd.merge(subject_data, visit_data, how='left', on='subject_id', validate='1:m')
    visit_data = visit_data.loc[~visit_data['neurobooth_visit_dates'].isna()]  # Exclude rows with no visit date

    # TODO: Fuzzy-join PROMs with visit-level data
    for table_spec, prom in prom_data.items():
        possible_matches = pd.merge(visit_data, prom, how='left', on='subject_id')

        delta_column = f'{table_spec.form_name}_days_offset'
        possible_matches[delta_column] = \
            (possible_matches[table_spec.end_time_column] - possible_matches['neurobooth_visit_dates'])
        possible_matches[delta_column] /= np.timedelta64(1, 'D')  # Convert to days

        # TODO: Was trying to do inline absolute value here...
        # Rank possible matches for each visit based on proximity of survey completion time
        # possible_matches['Rank'] = possible_matches.abs() \
        #     .groupby(['subject_id', 'neurobooth_visit_dates'])[delta_column] \
        #     .rank(method='min', na_option='bottom')

        print(possible_matches)

    # TODO: Create a list of contact information for people who have uncompleted surveys
    # TODO: Create a list of survey completion percentages for each subject
    # TODO: Create histograms + statistics for how long each survey takes


if __name__ == '__main__':
    main()
