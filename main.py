from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil, os, json
from fastapi.middleware.cors import CORSMiddleware

from pipeline_analysis import ProcessVideo

app = FastAPI()

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
    video_path = f"videos/{file.filename}"
    os.makedirs("videos", exist_ok=True)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ProcessVideo(video_path)

    return {"status": "uploaded"}


@app.get("/progress")
async def get_progress():
    global progress
    # Replace with actual pipeline progress value
    progress = min(progress + 10, 100)
    return {"value": progress}


@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = f"results/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}
