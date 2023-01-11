"""
Generate a list of patient report completion, alongside completion statistics.
"""

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Connection
from typing import NamedTuple, List
from neurobooth_reports.settings import ReportSettings


PROM_RC_FORMS: List[str] = [
    # 'handedness_questionnaire',  # Currently missing
    # 'chief_short_form',
    # 'communicative_participation_item_bank',  # Table is empty
    # 'dysarthria_impact_scale',
    # 'neuro_qol_anxiety_short_form',
    # 'neuro_qol_cognitive_function_short_form',
    # 'neuro_qol_depression_short_form',
    # 'neuro_qol_emotional_dyscontrol_short_form',  # Table is empty
    # 'neuro_qol_fatigue_short_form',
    # 'neuro_qol_le_short_form',
    # 'neuro_qol_participate_social_roles_short_form',  # Table is empty
    # 'neuro_qol_positive_affect_and_wellbeing_short_form',
    # 'neuro_qol_satisfaction_social_roles_short_form',  # Table is empty
    # 'neuro_qol_sleep_disturbance_short_form',
    # 'neuro_qol_stigma_short_form',  # Table is empty
    # 'neuro_qol_ue_short_form',
    'neurobooth_falls',
    # 'neurobooth_vision_prom_ataxia',
    'prom_ataxia',
    # 'promis_10',
    'study_feedback',
    'system_usability_scale',
]


class PromTableSpec(NamedTuple):
    table_name: str
    start_time_column: str
    end_time_column: str
    completion_column: str


def form_name_to_table_spec(form_name: str) -> PromTableSpec:
    """Take advantage of naming regularities to get Neurobooth table and column names from the Redcap form name."""
    return PromTableSpec(
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
        ],
    ).convert_dtypes()

    # Decode phone type
    contact_int = table['phone_type_contact'].astype('Int64').copy()
    table['phone_type_contact'] = '[Unknown]'
    table.loc[contact_int == 1, 'phone_type_contact'] = 'Mobile'
    table.loc[contact_int == 2, 'phone_type_contact'] = 'Landline'
    table['phone_type_contact'] = table['phone_type_contact'].astype('string')

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


def main():
    settings = ReportSettings()

    table_specs = list(map(form_name_to_table_spec, PROM_RC_FORMS))

    engine = create_engine(settings.database_connection_info.postgresql_url())
    with engine.connect() as connection:
        prom_data = {spec: download_prom_table(spec, connection) for spec in table_specs}
        subject_data = download_subject_table(connection)
        contact_data = download_contact_table(connection)
        visit_data = download_visit_dates(connection)

    # TODO: Fuzzy-join everything together!
    # TODO: Create a list of contact information for people who have uncompleted surveys
    # TODO: Create a list of survey completion percentages for each subjecy
    # TODO: Create histograms + statistics for how long each survey takes


if __name__ == '__main__':
    main()
