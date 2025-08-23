import time
from court_detection.main import GetCornerPoints
from ball_detection.main import produce_ball_coordinates
from apply_yolo.main import apply_YOLO
from apply_yolo.test import get_all_yolo_coords
from generate_heatmap.main import produce_heatmaps
import json

from helpers import update_progress


def ProcessVideo(video_path, task_id, progress_store):
    try:
        update_progress(progress_store, task_id, 10, "Finding Corners")
        corner_points = GetCornerPoints(video_path, task_id)

        update_progress(progress_store, task_id, 40, "Runnning YOLO")
        apply_YOLO(video_path, corner_points, progress_store, task_id)

        update_progress(progress_store, task_id, 70, "Generating Heatmaps")
        with open("playerTop.json", "r") as f:
            playerTop = json.load(f)
        with open("playerBottom.json", "r") as f:
            playerBottom = json.load(f)
        produce_heatmaps(
            playerTop=playerTop,
            playerBottom=playerBottom,
            progress_store=progress_store,
            bw_method=0.5,
            task_id=task_id,
        )

        update_progress(progress_store, task_id, 100, "Done")
    except Exception as e:
        update_progress(progress_store, task_id, -1, f"error: {str(e)}")


def main():
    video_path = "video_input2.mp4"
    ProcessVideo(video_path)


if __name__ == "__main__":
    main()
