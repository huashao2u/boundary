#!/usr/bin/env python3
"""Send an email to mario2zxy1234@gmail.com via SMTP.

Default behavior:
  Sends from songyanlong@iie.ac.cn to mario2zxy1234@gmail.com using CSTNet SMTP.

Optional environment variables:
  SMTP_HOST       SMTP server hostname, default mail.cstnet.cn
  SMTP_PORT       SMTP port, default 465
  SMTP_USERNAME   Sender email account, default songyanlong@iie.ac.cn
  SMTP_PASSWORD   SMTP password or client-specific password, default is embedded below
  MAIL_FROM       Sender address, default SMTP_USERNAME

Warning:
  This script contains an embedded email password for one-click execution.
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage


DEFAULT_RECIPIENT = "mario2zxy1234@gmail.com"
DEFAULT_SENDER = "songyanlong@iie.ac.cn"
DEFAULT_SMTP_HOST = "mail.cstnet.cn"
DEFAULT_SMTP_PORT = 465
DEFAULT_SMTP_PASSWORD = "N4e?~YQf^&IjtH8!"


def build_message(sender: str, recipient: str, subject: str, body: str) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return message


def send_email(message: EmailMessage, host: str, port: int, username: str, password: str) -> None:
    context = ssl.create_default_context()
    if port in {465, 994}:
        server: smtplib.SMTP = smtplib.SMTP_SSL(host, port, timeout=30, context=context)
    else:
        server = smtplib.SMTP(host, port, timeout=30)

    with server:
        if port not in {465, 994}:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
        server.login(username, password)
        server.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Send an email to {DEFAULT_RECIPIENT}.")
    parser.add_argument("--to", default=DEFAULT_RECIPIENT, help="Recipient email address.")
    parser.add_argument("--smtp-host", default=os.getenv("SMTP_HOST", DEFAULT_SMTP_HOST), help="SMTP server host.")
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.getenv("SMTP_PORT", str(DEFAULT_SMTP_PORT))),
        help="SMTP server port.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("SMTP_USERNAME", DEFAULT_SENDER),
        help="SMTP login username.",
    )
    parser.add_argument("--subject", default="Hello from Python", help="Email subject.")
    parser.add_argument(
        "--body",
        default="This email was sent by /media/boundary/send_email.py.",
        help="Plain text email body.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = os.getenv("SMTP_PASSWORD", DEFAULT_SMTP_PASSWORD)
    sender = os.getenv("MAIL_FROM", args.username)

    message = build_message(sender, args.to, args.subject, args.body)
    try:
        send_email(message, args.smtp_host, args.smtp_port, args.username, password)
    except smtplib.SMTPAuthenticationError as exc:
        raise SystemExit(
            "SMTP authentication failed. Check the account password, or use the CSTNet "
            "client-specific password if this mailbox requires one."
        ) from exc
    print(f"Email sent to {args.to}")


if __name__ == "__main__":
    main()
