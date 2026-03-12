import boto3
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )


# ==========================================================
# Upload PDF
# ==========================================================
def upload_pdf_to_s3(file_path):

    if not BUCKET_NAME:
        raise Exception("AWS_BUCKET_NAME is missing in .env")

    s3 = get_s3_client()

    filename = os.path.basename(file_path)

    s3.upload_file(
        file_path,
        BUCKET_NAME,
        filename,
        ExtraArgs={"ContentType": "application/pdf"}
    )

    print("PDF uploaded to S3")

    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"

    return url


# Upload Image
def upload_image_to_s3(file_path, scan_id):

    if not BUCKET_NAME:
        raise Exception("AWS_BUCKET_NAME is missing in .env")

    s3 = get_s3_client()

    key = f"images/{scan_id}.jpg"

    s3.upload_file(
        file_path,
        BUCKET_NAME,
        key,
        ExtraArgs={"ContentType": "image/jpeg"}
    )

    print("Image uploaded to S3")

    url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return url