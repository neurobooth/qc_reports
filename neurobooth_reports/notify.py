"""
Functions to send status emails to a list of people.
"""

import smtplib
import email

from neurobooth_reports.settings import ReportSettings


def send_emails(message: str, subject: str, settings: ReportSettings):
    msg = email.message.EmailMessage()
    msg['From'] = settings.email.from_addr
    msg['Subject'] = f'{settings.email.subject_prefix}{subject}'
    msg['To'] = ', '.join(settings.email.to_addr)
    msg.set_content(message)

    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)
