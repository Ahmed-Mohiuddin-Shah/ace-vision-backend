import ultralytics
import cv2
import json

model = ultralytics.YOLO("apply_yolo/yolov8n.pt")  # pretrained YOLOv8n model

dictionary = {
    "topLeft": (577, 302),
    "topRight": (1335, 302),
    "bottomLeft": (362, 859),
    "bottomRight": (1567, 859),
}


topLeft = dictionary["topLeft"]
topRight = dictionary["topRight"]
bottomLeft = dictionary["bottomLeft"]
bottomRight = dictionary["bottomRight"]


def euclidean_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


results = None


def apply_yolo(video_path):
    global results
    results = model(
        video_path, classes=[0], verbose=False, stream=True
    )  # return a list of Results objects

    return results


def get_all_yolo_coords():

    coords = []
    i = 0
    while True:
        try:
            for frame_number, r in enumerate(results):
                coords.append(get_yolo_coords(r, 1080, 1920))
            # coords.append(get_yolo_coords(i, 1080, 1920))
        except:
            break

        i += 1

    return coords


def get_yolo_coords(frame_result, frame_height, frame_width):
    coords = frame_result.boxes.xyxyn.cpu().numpy()

    # print("frame_height : ", frame_height)
    # print("frame_width : ", frame_width)

    minTop = None
    minBottom = None

    minTopDist = 10000
    minBottomDist = 10000

    for coord in coords:
        x1 = int(coord[0] * frame_width)
        y1 = int(coord[1] * frame_height)
        x2 = int(coord[2] * frame_width)
        y2 = int(coord[3] * frame_height)

        avgDist = (
            euclidean_distance(x1, y1, topLeft[0], topLeft[1])
            + euclidean_distance(x1, y1, topRight[0], topRight[1])
        ) / 2
        if avgDist < minTopDist:
            minTopDist = avgDist
            minTop = [x1, y1, x2, y2]

        avgDist = (
            euclidean_distance(x2, y2, bottomLeft[0], bottomLeft[1])
            + euclidean_distance(x2, y2, bottomRight[0], bottomRight[1])
        ) / 2
        if avgDist < minBottomDist:
            minBottomDist = avgDist
            minBottom = [x1, y1, x2, y2]

            # append_text_to_file("results.txt", f'index : ${str(index)} minTop: {str(minTop)} minBottom: {str(minBottom)}')

    # minTop[0] -= 10
    # minTop[1] -= 10
    # minTop[2] += 10
    # minTop[3] += 10

    # minBottom[0] -= 10
    # minBottom[1] -= 10
    # minBottom[2] += 10
    # minBottom[3] += 10

    return minTop, minBottom


# cv2.rectangle(frame, (minTop[0], minTop[1]), (minTop[2], minTop[3]), (0, 0, 255), 2)
# cv2.rectangle(frame, (minBottom[0], minBottom[1]), (minBottom[2], minBottom[3]), (0, 0, 255), 2)

# cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
# cv2.line(frame, topLeft, bottomLeft, (0, 255, 0), 2)
# cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
# cv2.line(frame, bottomLeft, bottomRight, (0, 255, 0), 2)

# top_image = frame1[minTop[1]:minTop[3], minTop[0]:minTop[2]]
# bottom_image = frame1[minBottom[1]:minBottom[3], minBottom[0]:minBottom[2]]

# cv2.imwrite("top.jpg", top_image)
# cv2.imwrite("bottom.jpg", bottom_image)


# cv2.imwrite("frame.jpg", frame)

# json_string = json.dumps({"top": minTop, "bottom": minBottom})
# with open("data.json", "w") as json_file:
#     json_file.write(json_string)
