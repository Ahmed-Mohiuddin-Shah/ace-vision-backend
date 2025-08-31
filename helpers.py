import json
import time
from PIL import Image
import os
import cv2
import numpy as np


def save_frame_pil(frame, path):
    # Convert OpenCV BGR → RGB
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    with open(path, "wb") as f:
        img.save(f, format="png")
        f.flush()
        os.fsync(f.fileno())  # force flush to disk


def update_progress(progress_store, task_id, percent, status):
    task = progress_store.get(task_id, {})
    start_time = task.get("start_time", time.time())
    elapsed = time.time() - start_time

    if percent > 0:
        eta_seconds = (elapsed / percent) * (100 - percent)
        eta_time = time.time() + eta_seconds
        eta_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(eta_time))
    else:
        eta_seconds = None
        eta_iso = None

    progress_store[task_id] = {
        "progress": percent,
        "status": status,
        "start_time": start_time,
        "eta_seconds": round(eta_seconds, 2) if eta_seconds else None,
        "eta": eta_iso,
    }
    # update progress_store.json file
    with open("progress_store.json", "w") as f:
        json.dump(progress_store, f)
