import boto3
import os
from dotenv import load_dotenv

# ✅ Load .env variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def upload_pdf_to_s3(file_path):

    if not BUCKET_NAME:
        raise Exception("❌ AWS_BUCKET_NAME is missing in .env file")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )

    filename = os.path.basename(file_path)
    print("filename", filename)
    print("file_path", file_path)
    print("BUCKET_NAME", BUCKET_NAME)
    print("AWS_REGION", AWS_REGION)
    print("AWS_ACCESS_KEY", AWS_ACCESS_KEY)
    print("AWS_SECRET_KEY", AWS_SECRET_KEY)

    # Upload file
    s3.upload_file(
        file_path,
        BUCKET_NAME,
        filename,
        ExtraArgs={"ContentType": "application/pdf"}
    )
    print("uploaded...!")

    # Generate public URL
    url = f"https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{filename}"

    return url