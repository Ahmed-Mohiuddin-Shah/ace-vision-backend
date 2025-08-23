import time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import shutil, os, json, uuid, threading
from fastapi.middleware.cors import CORSMiddleware

from apply_yolo.main import apply_YOLO
from court_detection.main import GetCornerPoints
from generate_heatmap.main import produce_heatmaps
from pipeline_analysis import ProcessVideo


app = FastAPI()

# global progress store
progress_store = {}

# check if progress_store.json exists, if yes load it
if os.path.exists("progress_store.json"):
    with open("progress_store.json", "r") as f:
        if os.stat("progress_store.json").st_size != 0:
            progress_store = json.load(f)
else:
    with open("progress_store.json", "w") as f:
        json.dump(progress_store, f)

# Allow Next.js frontend (localhost:3000) to access
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],  # allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)
progress = 0


@app.post("/player-info")
async def save_player_info(data: dict):
    with open("player_info.json", "w") as f:
        json.dump(data, f, indent=4)
    return {"status": "saved"}


@app.get("/player-info")
async def get_player_info():
    if not os.path.exists("player_info.json"):
        return {
            "name": "",
            "age": "",
            "weight": "",
            "height": "",
            "gender": "",
            "dominant_hand": "",
        }

    with open("player_info.json", "r") as f:

        if os.stat("player_info.json").st_size == 0:
            return {
                "name": "",
                "age": "",
                "weight": "",
                "height": "",
                "gender": "",
                "dominant_hand": "",
            }

        player_info = json.load(f)

        return player_info


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    global progress_store
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create unique task id
    task_id = str(uuid.uuid4())
    progress_store[task_id] = {
        "progress": 0,
        "status": "starting",
        "start_time": time.time(),
        "eta": None,
    }
    
    # save progress_store to json file
    with open("progress_store.json", "w") as f:
        json.dump(progress_store, f)

    # Run process in background
    thread = threading.Thread(
        target=ProcessVideo, args=(video_path, task_id, progress_store)
    )
    thread.start()

    return {"status": "uploaded", "task_id": task_id}


@app.get("/progress/{task_id}")
def get_progress(task_id: str):
    return progress_store.get(task_id, {"progress": 0, "status": "unknown"})


@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = f"results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}


@app.get("/results")
async def get_results():
    # get task_ids from progress_store
    task_ids = list(progress_store.keys())
    results = {}
    for task_id in task_ids:
        # get progress for each task_id if running else get the result file
        if progress_store[task_id]["progress"] < 100:
            results[task_id] = {
                "progress": progress_store[task_id]["progress"],
                "status": progress_store[task_id]["status"],
            }
        else:
            # get the result file
            result_file = f"results/minimap_heatmap_{task_id}.png"
            if os.path.exists(result_file):
                results[task_id] = {
                    "progress": 100,
                    "status": "completed",
                    "result_file": result_file,
                }
            else:
                results[task_id] = {
                    "progress": 100,
                    "status": "completed",
                    "result_file": None,
                }
    return results
