
from court_detection.detection import generate_corner_points, refine_corner_points, get_line_coordinates
import cv2
import numpy as np
import time


input_path = 'frame.jpg'
output_path = 'output.jpg'

def main():
    
    start_time = time.time()

    # for image
    image = cv2.imread(input_path)

    image_copy = np.copy(image)


    mask = (image < 182).any(axis=-1)
    image[mask] = 0

    cv2.imwrite('mask.jpg', image)

    dictionary = generate_corner_points(np.copy(image))

    image1 = np.copy(image)

    cv2.circle(image1, dictionary['topLeft'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['topRight'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['bottomLeft'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['bottomRight'], 5, (0, 255, 0), -1)

    cv2.imwrite("corner_points.jpg", image1)

    dictionary_refined = refine_corner_points(dictionary, kernel_size=35, image=np.copy(image), threshold=230)

    print("FINAL DICTIONARY : ", dictionary_refined)

    color = (0, 0, 255)

    coordinates = get_line_coordinates(dictionary_refined, np.copy(image), 8, color=color)

    for c in coordinates: 
        image_copy[c[0], c[1]] = color

    cv2.imwrite(output_path, image_copy)

    print("Overall Time taken: ", time.time() - start_time)

def GetCornerPoints(input_path_video):
    # get a random frame from video

    cap = cv2.VideoCapture(input_path_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()

    dictionary = generate_corner_points(np.copy(frame))

    try: 
        dictionary_refined = refine_corner_points(dictionary, kernel_size=35, image=np.copy(frame), threshold=230)
    except: 
        dictionary_refined = dictionary

    print("FINAL DICTIONARY : ", dictionary_refined)

    #plot the corner points on the image
    image1 = np.copy(frame)

    cv2.circle(image1, dictionary['topLeft'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['topRight'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['bottomLeft'], 5, (0, 255, 0), -1)
    cv2.circle(image1, dictionary['bottomRight'], 5, (0, 255, 0), -1)

    cv2.imwrite("results/corner_points.jpg", image1)

    return dictionary

if __name__ == '__main__':
    main()