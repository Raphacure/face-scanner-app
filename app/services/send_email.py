import os
import base64
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = "wellness@raphacure.com"
DEFAULT_BCC_EMAILS = ["maillogs@raphacure.com"]


def download_file(url, file_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def send_email(
    to,
    subject,
    html,
    file_urls=None,
    cc=None,
    bcc=None,
    from_email=None
):

    if not to or not subject:
        raise ValueError("Missing required fields: to, subject")

    file_urls = file_urls or []
    cc = cc or []
    bcc = bcc or []

    bcc_emails = DEFAULT_BCC_EMAILS + bcc
    attachments = []
    temp_files = []

    try:

        # download files
        for index, url in enumerate(file_urls):
            temp_file = f"temp_report_{index}.pdf"
            print("Downloading:", url)

            download_file(url, temp_file)

            with open(temp_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()

            attachment = Attachment(
                FileContent(encoded),
                FileName(f"report_{index+1}.pdf"),
                FileType("application/pdf"),
                Disposition("attachment")
            )

            attachments.append(attachment)
            temp_files.append(temp_file)

        message = Mail(
            from_email=from_email or FROM_EMAIL,
            to_emails=to,
            subject=subject,
            html_content=html
        )

        # add cc
        if cc:
            message.cc = cc

        # add bcc
        if bcc_emails:
            message.bcc = bcc_emails

        # add attachments
        for att in attachments:
            message.add_attachment(att)

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        print("Email sent:", response.status_code)

        return {"success": True}

    except Exception as e:
        print("Error sending email:", str(e))
        raise

    finally:
        # cleanup temp files
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)