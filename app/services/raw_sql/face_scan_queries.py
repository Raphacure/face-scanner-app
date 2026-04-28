import json
from typing import Any, Dict, Optional

from app.db import run_query


def insert_face_scan_record(scan_data: Dict[str, Any]) -> None:
    device = scan_data.get("device")
    ip = scan_data.get("ip")
    user_id = scan_data.get("user_id")
    client_id = scan_data.get("client_id")
    report_url = scan_data.get("report_url")
    image_url = scan_data.get("image_url")
    response = scan_data.get("response") or {}

    response_json: Optional[str]
    try:
        response_json = json.dumps(response)
    except TypeError:
        response_json = json.dumps({"raw": str(response)})

    if report_url is None:
        report_url_values: Optional[list[str]] = None
    elif isinstance(report_url, list):
        report_url_values = [str(url) for url in report_url]
    else:
        report_url_values = [str(report_url)]

    run_query(
        """
        INSERT INTO face_scan (device, ip, user_id, client_id, report_url, image_url, response)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        """,
        (device, ip, user_id, client_id, report_url_values, image_url, response_json),
    )

