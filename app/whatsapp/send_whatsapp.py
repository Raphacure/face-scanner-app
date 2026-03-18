import requests
import os


def send_whatsapp_pdf(to: str, pdf_url1: str, pdf_url2: str):

    access_token = os.getenv("WA_TOKEN")
    phone_number_id = os.getenv("WA_PHONENUMBER_ID")

    url = f"https://graph.facebook.com/v13.0/{phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    try:

        # First PDF
        payload1 = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "document",
            "document": {
                "link": pdf_url1,
                "caption": "Here is your face scan report.",
                "filename": "Face_Scan_Report_v1.pdf"
            }
        }

        response1 = requests.post(url, json=payload1, headers=headers)
        response1.raise_for_status()


        # Second PDF
        payload2 = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "document",
            "document": {
                "link": pdf_url2,
                "caption": "Here is your advanced face scan report.",
                "filename": "Face_Scan_Report_v2.pdf"
            }
        }

        response2 = requests.post(url, json=payload2, headers=headers)
        response2.raise_for_status()

        print("✅ Both PDFs sent successfully")

        return {
            "report_v1": response1.json(),
            "report_v2": response2.json()
        }

    except requests.exceptions.RequestException as error:
        print("❌ Error sending PDF:", error)
        raise