import time


def update_progress(progress_store, task_id, percent, status):
    task = progress_store.get(task_id, {})
    start_time = task.get("start_time", time.time())
    elapsed = time.time() - start_time

    if percent > 0:
        eta = (elapsed / percent) * (100 - percent)
    else:
        eta = None

    progress_store[task_id] = {
        "progress": percent,
        "status": status,
        "start_time": start_time,
        "eta": round(eta, 2) if eta else None,
    }
