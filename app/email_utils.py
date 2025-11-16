import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
from datetime import datetime

def clean_html(raw_html: str) -> str:
    """Convert HTML email body to clean text."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text("\n", strip=True)


def decode_header_value(value):
    """Decode MIME encoded headers."""
    if value is None:
        return ""
    decoded, encoding = decode_header(value)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(encoding or "utf-8", errors="ignore")
    return decoded


def fetch_emails_between_dates(email_user, email_pass, start_date, end_date):
    """
    Fetch emails between two dates using IMAP.
    Dates should be 'YYYY-MM-DD'
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")

    # Login
    mail.login(email_user, email_pass)

    mail.select("inbox")

    # Convert to DD-MMM-YYYY format for IMAP search
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%b-%Y")
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%b-%Y")

    # IMAP search query
    status, data = mail.search(None, f'(SINCE "{start}" BEFORE "{end}")')

    if status != "OK":
        return []

    email_ids = data[0].split()
    emails = []

    for eid in email_ids:
        status, message_data = mail.fetch(eid, "(RFC822)")
        if status != "OK":
            continue

        msg = email.message_from_bytes(message_data[0][1])

        subject = decode_header_value(msg["Subject"])
        sender = decode_header_value(msg["From"])
        date = msg["Date"]

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    body = part.get_payload(decode=True).decode(errors="ignore")
                    break
                elif ctype == "text/html":
                    html = part.get_payload(decode=True).decode(errors="ignore")
                    body = clean_html(html)
                    break
        else:
            if msg.get_content_type() == "text/plain":
                body = msg.get_payload(decode=True).decode(errors="ignore")
            else:
                html = msg.get_payload(decode=True).decode(errors="ignore")
                body = clean_html(html)

        emails.append({
            "subject": subject,
            "sender": sender,
            "date": date,
            "body": body
        })

    mail.logout()
    return emails
