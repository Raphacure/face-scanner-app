import requests
import os


def send_whatsapp_pdf(to: str, pdf_url: str):
    """
    Send a PDF document via WhatsApp Cloud API.

    Args:
        to (str): Receiver phone number with country code (ex: 919876543210)
        pdf_url (str): Public PDF file URL

    Returns:
        dict: WhatsApp API response
    """

    access_token = os.getenv("WA_TOKEN")
    phone_number_id = os.getenv("WA_PHONENUMBER_ID")

    url = f"https://graph.facebook.com/v13.0/{phone_number_id}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": {
            "link": pdf_url,
            "caption": "Here is your report.",
            "filename": "Health_Report.pdf"
        }
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        print("✅ PDF sent successfully:", response.json())
        return response.json()

    except requests.exceptions.RequestException as error:
        print("❌ Error sending PDF:", error)
        if response is not None:
            print("Response:", response.text)
        raise
