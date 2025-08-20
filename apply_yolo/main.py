import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
from scipy.interpolate import CubicSpline

# read in the data from the json files
# with open('coordinates.json') as f:
#     data = json.load(f)

start_idx = 0


def Interpolate_points(x_coordinates, y_coordinates):
    global start_idx

    # # extract the x and y coordinates
    # x_coordinates = coordinates['x_coordinates']
    # y_coordinates = coordinates['y_coordinates']

    # Find the first non-None value for x and y
    first_non_none_x = next(
        (i for i, x in enumerate(x_coordinates) if x is not None), None
    )
    first_non_none_y = next(
        (i for i, y in enumerate(y_coordinates) if y is not None), None
    )

    # Determine the start index (max of the two indices)
    start_idx = max(first_non_none_x, first_non_none_y)

    # Trim the lists to start from the first non-None value
    x_coordinates = x_coordinates[start_idx:]
    y_coordinates = y_coordinates[start_idx:]

    print("Coordinates before interpolation:", x_coordinates, y_coordinates)

    coordinates = []

    for i in range(len(x_coordinates)):
        x = float(x_coordinates[i]) if x_coordinates[i] is not None else np.nan
        y = float(y_coordinates[i]) if y_coordinates[i] is not None else np.nan
        coordinates.append([x, y])

    coordinates = np.array(coordinates)

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    t = np.arange(len(coordinates))

    # Filter out NaN values for spline fitting
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    t_valid = t[valid_indices]
    x_valid = x[valid_indices]
    y_valid = y[valid_indices]

    # Use CubicSpline for interpolation
    x_spline = CubicSpline(t_valid, x_valid)
    y_spline = CubicSpline(t_valid, y_valid)

    # Interpolate over the original range
    x_interpolated = x_spline(t)
    y_interpolated = y_spline(t)

    print("Coordinates after interpolation:", x_interpolated, y_interpolated)

    # Smoothing the data
    x_smoothed = scipy.ndimage.gaussian_filter1d(x_interpolated, sigma=1)
    y_smoothed = scipy.ndimage.gaussian_filter1d(y_interpolated, sigma=1)

    print("Coordinates after smoothing:", x_smoothed, y_smoothed)

    return x_smoothed, y_smoothed


from apply_yolo.test import apply_yolo, get_yolo_coords
from ball_detection import *
import cv2
import numpy as np


class CourtConfigurations:
    def __init__(self) -> None:
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)
        self.line_width = 1
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2

    def build_court_reference(self):
        """
        Create court reference image using the lines positions
        """
        court = np.zeros(
            (
                self.court_height + 2 * self.top_bottom_border,
                self.court_width + 2 * self.right_left_border,
            ),
            dtype=np.uint8,
        )
        cv2.line(court, *self.baseline_top, (255, 255, 255), self.line_width)
        cv2.line(court, *self.baseline_bottom, (255, 255, 255), self.line_width)
        cv2.line(court, *self.net, (255, 255, 255), self.line_width)
        cv2.line(court, *self.top_inner_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.bottom_inner_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.left_court_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.right_court_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.left_inner_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.right_inner_line, (255, 255, 255), self.line_width)
        cv2.line(court, *self.middle_line, (255, 255, 255), self.line_width)
        court = cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))
        # cv2.imwrite('court_reference.png', court)

        return court


CourtObj = CourtConfigurations()
court_reference = CourtObj.build_court_reference()
width_court_frame = court_reference.shape[1]
height_court_frame = court_reference.shape[0]

court_frame = cv2.imread("apply_yolo/minimap.png")


def apply_YOLO(dictionary):

    src_points = np.float32(
        [
            dictionary["topLeft"],
            dictionary["topRight"],
            dictionary["bottomLeft"],
            dictionary["bottomRight"],
        ]
    )

    dst_points = np.float32(
        [
            CourtObj.baseline_top[0],
            CourtObj.baseline_top[1],
            CourtObj.baseline_bottom[0],
            CourtObj.baseline_bottom[1],
        ]
    )

    M = cv2.getPerspectiveTransform(src_points, dst_points)


def generate_ball_plottings(
    frame_width, frame_height, x_coordinates, y_coordinates, file_name
):

    img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    for x, y in zip(x_coordinates, y_coordinates):
        if x != None and y != None:
            try:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
            except:
                pass

    cv2.imwrite(file_name, img)


playerTop = []
playerBottom = []


def map_to_minimap(pt, M):
    print("starting")

    player_coordinates = np.array([[pt[0], pt[1], 1]], dtype=np.float32)
    mapped_coordinates = np.dot(M, player_coordinates.T).T
    mapped_x, mapped_y = mapped_coordinates[0, :2] / mapped_coordinates[0, 2]

    print("ending")
    return (mapped_x, mapped_y)


def apply_YOLO(video_input_path, dictionary):

    src_points = np.float32(
        [
            dictionary["topLeft"],
            dictionary["topRight"],
            dictionary["bottomLeft"],
            dictionary["bottomRight"],
        ]
    )

    dst_points = np.float32(
        [
            CourtObj.baseline_top[0],
            CourtObj.baseline_top[1],
            CourtObj.baseline_bottom[0],
            CourtObj.baseline_bottom[1],
        ]
    )

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    results = apply_yolo(video_input_path)

    # x_smoothed, y_smoothed = Interpolate_points(x_coordinates, y_coordinates)

    # print("len(x_coordinates): ", len(x_coordinates), "len(y_coordinates): ", len(y_coordinates))
    # print("len(x_smoothed): ", len(x_smoothed), "len(y_smoothed): ", len(y_smoothed))

    # return

    # x_sharp_coordinates, y_sharp_coordinates = get_sharp_points(x_smoothed, y_smoothed)

    # combined_sharp_coordinates = list(zip(x_sharp_coordinates, y_sharp_coordinates))

    # for i in range(start_idx):
    #     x_smoothed = np.insert(x_smoothed, 0, None)
    #     y_smoothed = np.insert(y_smoothed, 0, None)
    #     # x_smoothed.insert(0, None)
    #     # y_smoothed.insert(0, None)

    # generate_ball_plottings(1920, 1080, x_smoothed, y_smoothed, "smoothed_0.png")

    # combined_smoothed_coordinates = list(zip(x_smoothed, y_smoothed))

    cap = cv2.VideoCapture(video_input_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
    # output_video_path = 'output.avi'
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # court_frame = cv2.imread("copied_frame.png")

    # out_minimap = cv2.VideoWriter('minimap.avi', fourcc, fps, (width_court_frame, height_court_frame))

    court_frame = cv2.imread("apply_yolo/minimap.png")

    transformed_sharp_coordinates = []

    # for coord in combined_sharp_coordinates:
    #     new_coord = map_to_minimap(coord)
    #     transformed_sharp_coordinates.append(new_coord)

    #     cv2.circle(court_frame, (int(new_coord[0]), int(new_coord[1])), 10, (0, 255, 0), -1)

    # cv2.imwrite("minimap.png", court_frame)

    # save transformed_sharp_coordinatesto a json file
    # with open('transformed_sharp_coordinates.json', 'w') as f:
    #     json.dump(transformed_sharp_coordinates, f)

    def midpoint_of_bottom_line(topLeft, bottomRight):
        x1, y1 = topLeft
        x2, y2 = bottomRight

        midpoint_x = (x1 + x2) / 2
        midpoint_y = y2  # Since it's the bottom line, the y-coordinate remains the same

        return (midpoint_x, midpoint_y)

    for frame_number, r in enumerate(results):
        ret, frame = cap.read()

        if not ret:
            break

        coords = get_yolo_coords(
            r, frame_height=frame_size[1], frame_width=frame_size[0]
        )
        print("coords: ", coords)

        try:
            # get midpoint of bottomLeft and bottomRight coordinates for both players
            midBottomTop = midpoint_of_bottom_line(
                (coords[0][0], coords[0][1]), (coords[0][2], coords[0][3])
            )
            midBottomTop = map_to_minimap(midBottomTop, M)
            playerTop.append(midBottomTop)

            midBottomBottom = midpoint_of_bottom_line(
                (coords[1][0], coords[1][1]), (coords[1][2], coords[1][3])
            )
            midBottomBottom = map_to_minimap(midBottomBottom, M)
            playerBottom.append(midBottomBottom)

            # plot on copied_frame.png
            cv2.circle(
                court_frame,
                (int(midBottomTop[0]), int(midBottomTop[1])),
                10,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                court_frame,
                (int(midBottomBottom[0]), int(midBottomBottom[1])),
                10,
                (0, 255, 0),
                -1,
            )

        except:
            pass

        # Break the loop if 'q' is pressed
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    if court_frame is None or court_frame.size == 0:
        print("No frame captured or frame is empty")
    else:
        cv2.imwrite("results/minimap_heatmap.png", court_frame)

    cap.release()
    # out.release()
    # out_minimap.release()
    cv2.destroyAllWindows()

    # save playerTop and playerBottom to a json file
    with open("playerTop.json", "w") as f:
        json.dump(playerTop, f)

    with open("playerBottom.json", "w") as f:
        json.dump(playerBottom, f)
