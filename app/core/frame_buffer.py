from collections import defaultdict
import time

BUFFERS = defaultdict(list)
TIMESTAMPS = {}

MAX_FRAMES = 100
TIMEOUT_SECONDS = 30


def add_frame(scan_id, frame):
    BUFFERS[scan_id].append(frame)
    TIMESTAMPS[scan_id] = time.time()


def is_ready(scan_id):
    return len(BUFFERS[scan_id]) >= MAX_FRAMES


def get_frames(scan_id):
    return BUFFERS[scan_id]


def count(scan_id):
    return len(BUFFERS[scan_id])


def clear(scan_id):
    BUFFERS.pop(scan_id, None)
    TIMESTAMPS.pop(scan_id, None)


def cleanup_expired():
    now = time.time()
    expired = [
        sid for sid, ts in TIMESTAMPS.items()
        if now - ts > TIMEOUT_SECONDS
    ]
    for sid in expired:
        clear(sid)
