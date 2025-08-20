from court_detection.main import GetCornerPoints
from ball_detection.main import produce_ball_coordinates
from apply_yolo.main import apply_YOLO
from apply_yolo.test import get_all_yolo_coords
from generate_heatmap.main import produce_heatmaps
import json



def ProcessVideo(video_path):
    corner_points = GetCornerPoints(video_path)

    print("Corner points: ", corner_points)

    # x_coordinates, y_coordinates = produce_ball_coordinates(video_path)

    # print("X coordinates: ", x_coordinates)
    # print("Y coordinates: ", y_coordinates)

    apply_YOLO(video_path, corner_points)

    # yolo_coords = get_all_yolo_coords()

    # print("YOLO coordinates: ", yolo_coords)

    with open("playerTop.json", "r") as f:
        playerTop = json.load(f)

    with open("playerBottom.json", "r") as f:
        playerBottom = json.load(f)

    produce_heatmaps(playerTop=playerTop, playerBottom=playerBottom, bw_method=0.5)


def main():
    video_path = "video_input2.mp4"
    ProcessVideo(video_path)


if __name__ == '__main__':
    main() 