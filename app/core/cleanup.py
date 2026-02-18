import time
from app.core.frame_buffer import buffer, clear, SCAN_TIMEOUT


def cleanup_stale_scans():
    now = time.time()
    stale_ids = []

    for scan_id, data in list(buffer.items()):
        if now - data["last_update"] > SCAN_TIMEOUT:
            stale_ids.append(scan_id)

    for scan_id in stale_ids:
        print("🧹 Clearing stale scan:", scan_id)
        clear(scan_id)
