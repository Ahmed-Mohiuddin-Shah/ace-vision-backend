import cv2
from ball_detection.BallDetection import BallDetector
import numpy as np


def main():
    # Replace 'path_to_input_video.mp4' with the path to your input video file
    input_video_path = 'video_input3.mp4'

    ball_detector = BallDetector('Weights.pth', out_channels=2)
    last_seen = None
    
    # Create a video capture object for input video
    cap = cv2.VideoCapture(input_video_path)

    # Check if the input video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    # Get the video frame dimensions and frame rate
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    black_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Define the codec and create a VideoWriter object to write the output video
    # Replace 'output_video.mp4' with the desired name of your output video file
    # uncomment below code to save the video
    # output_video_path = 'output_video.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)

    counter = 0

    while True:
        # Read a frame from the input video
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Process the frame
        ball_detector.detect_ball(frame)

        

        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            lastSeen = counter

        counter += 1

        # print("xy_coordinates", ball_detector.xy_coordinates.shape)
        # print("xy_coordinates[-1]", ball_detector.xy_coordinates[-1])

        
       
        # Write the processed frame to the output video
        # out.write(processed_frame)

        # # Display the processed frame in a window
        # cv2.imshow('Processed Frame', processed_frame)

        # Check if the user pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("FINAL xy_coordinates", ball_detector.xy_coordinates)

    # Release the video capture object and the output video writer
    cap.release()
    # out.release()

    # Close the display window
    cv2.destroyAllWindows()

    prev = None

    for coordinates in ball_detector.xy_coordinates:
        if prev is None and coordinates[0] is not None:
            prev = coordinates
            continue

        if coordinates[0] is not None:
            cv2.line(black_image, (prev[0], prev[1]), (coordinates[0], coordinates[1]), (0, 255, 0), 2)
            prev = coordinates

    cv2.imwrite("output_image.png", black_image)

def produce_ball_coordinates(video_path):
    ball_detector = BallDetector('ball_detection/Weights.pth', out_channels=2)
    last_seen = None
    
    # Create a video capture object for input video
    cap = cv2.VideoCapture(video_path)

    # Check if the input video file was opened successfully
    if not cap.isOpened():
        print("Error: Unable to open input video.")
        return

    counter = 0

    while True:
        # Read a frame from the input video
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Process the frame
        ball_detector.detect_ball(frame)

        if ball_detector.xy_coordinates[-1][0] is not None:
            ball = ball_detector.xy_coordinates[-1]
            lastSeen = counter

        counter += 1

    prev = None

    #extract and save x and y coordinates separately in json files:
    x_coordinates = []
    y_coordinates = []

    for coordinates in ball_detector.xy_coordinates:
        if prev is None and coordinates[0] is not None:
            prev = coordinates
            continue

        if coordinates[0] is not None:
            x_coordinates.append(coordinates[0])
            y_coordinates.append(coordinates[1])
            prev = coordinates

    return x_coordinates, y_coordinates

if __name__ == "__main__":
    main()
