import os

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

# ✅ IMPORT CLEANUP
from app.core.frame_buffer import cleanup_expired


app = FastAPI(title="Face Scan AI Service")

origins = [
    "http://localhost:3000",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ✅ BACKGROUND CLEANUP LOOP
@app.on_event("startup")
def start_cleanup_loop():

    def cleanup_loop():
        while True:
            cleanup_expired()
            time.sleep(5)   # every 5 seconds (good balance)

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


@app.get("/")
def home():
    return {"status": "AI Service Running"}
