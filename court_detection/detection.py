import cv2 
import numpy as np
import os
import court_detection.matrix_operations as MO
import time

# current_time_seconds = time.time()

# requires image to be in array format
def generate_corner_points(image):

    start_time = time.time()

    # image = cv2.imread(input_path)

    # get width and height of image
    height, width = image.shape[:2]

    # extra length for axis
    extraLen = width/3

    # axis with reference to the frame
    class axis:
        top = [[-extraLen,0],[width+extraLen,0]]
        right = [[width+extraLen,0],[width+extraLen,height]]
        bottom = [[-extraLen,height],[width+extraLen,height]]
        left = [[-extraLen,0],[-extraLen,height]]

    # creating a black image for mapping of courtlines
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Applying filters to the image for better detection
    gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gry, 156, 255, cv2.THRESH_BINARY)[1]
    canny = cv2.Canny(bw, 100, 200)

    # using hough lines probabilistic to detect lines with most intersections
    hPlines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=150, minLineLength=100, maxLineGap=10)

    frame_copy = cv2.imread('frame.jpg')
    for line in hPlines:
        cv2.line(frame_copy, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), thickness=2)

    # cv2.imwrite("hough_lines.png", frame_copy)

    intersectNum = np.zeros((len(hPlines),2))

    i = 0

    # Using nested loop to compare each line with every other line
    # to find the number of intersections
    for hPline1 in hPlines:
        Line1x1, Line1y1, Line1x2, Line1y2 = hPline1[0]
        Line1 = [[Line1x1,Line1y1],[Line1x2,Line1y2]]

        for hPline2 in hPlines:
            # if hPline1 is hPline2:
            #     continue

            Line2x1, Line2y1, Line2x2, Line2y2 = hPline2[0]
            Line2 = [[Line2x1,Line2y1],[Line2x2,Line2y2]]

            if Line1 is Line2: continue

            if Line1x1>Line1x2:
                temp = Line1x1
                Line1x1 = Line1x2
                Line1x2 = temp
                
            if Line1y1>Line1y2:
                temp = Line1y1
                Line1y1 = Line1y2
                Line1y2 = temp

            var = 200
            intersect = MO.findIntersection(Line1, Line2, Line1x1-200, Line1y1-200, Line1x2+var, Line1y2+var)

            if intersect is not None:
                intersectNum[i][0] += 1

        intersectNum[i][1] = i
        i += 1

    # Lines with most intersections get a fill mask command on them
    i = p = 0
    dilation = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)

    # cv2.imwrite("dilation.png", dilation)

    nonRectArea = dilation.copy()

    # sorting intersectNum with respect to first column in descending order
    intersectNum = intersectNum[(-intersectNum)[:, 0].argsort()]

    # filling the lines with most intersections
    for hPLine in hPlines:
        x1,y1,x2,y2 = hPLine[0]
        for p in range(8):
            if (i==intersectNum[p][1]) and (intersectNum[i][0]>0):
                cv2.floodFill(nonRectArea, np.zeros((height+2, width+2), np.uint8), (x1, y1), 1)
                cv2.floodFill(nonRectArea, np.zeros((height+2, width+2), np.uint8), (x2, y2), 1)
        i+=1

    dilation[np.where(nonRectArea == 255)] = 0

    dilation[np.where(nonRectArea == 1)] = 255

    eroded = cv2.erode(dilation, np.ones((5, 5), np.uint8))

    cannyMain = cv2.Canny(eroded, 90, 100)

    # cv2.imwrite("cannyoutput.png", cannyMain)

    # Extreme lines found every frame
    xOLeft = width + extraLen
    xORight = 0 - extraLen
    xFLeft = width + extraLen
    xFRight = 0 - extraLen

    yOTop = height
    yOBottom = 0
    yFTop = height
    yFBottom = 0

    # Finding all lines then allocate them to specified extreme variables
    # image = cv2.imread('frame.jpg')
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    hLines = cv2.HoughLines(cannyMain, 2, np.pi/180, 200)
    for hLine in hLines:
        for rho,theta in hLine:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + width*(-b))
            y1 = int(y0 + width*(a))
            x2 = int(x0 - width*(-b))
            y2 = int(y0 - width*(a))

            cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
            
            # Furthest intersecting point at every axis calculations done here
            intersectxF = MO.findIntersection(axis.bottom, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyO = MO.findIntersection(axis.left, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectxO = MO.findIntersection(axis.top, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            intersectyF = MO.findIntersection(axis.right, [[x1,y1],[x2,y2]], -extraLen, 0, width+extraLen, height)
            
            # if it even touches one axis line, code proceeds forward
            if (intersectxO is None) and (intersectxF is None) and (intersectyO is None) and (intersectyF is None):
                continue
            
            if intersectxO is not None:
                if intersectxO[0] < xOLeft:
                    xOLeft = intersectxO[0]
                    xOLeftLine = [[x1,y1],[x2,y2]]
                if intersectxO[0] > xORight:
                    xORight = intersectxO[0]
                    xORightLine = [[x1,y1],[x2,y2]]
            if intersectyO is not None:
                if intersectyO[1] < yOTop:
                    yOTop = intersectyO[1]
                    yOTopLine = [[x1,y1],[x2,y2]]
                if intersectyO[1] > yOBottom:
                    yOBottom = intersectyO[1]
                    yOBottomLine = [[x1,y1],[x2,y2]]
                    
            if intersectxF is not None:
                if intersectxF[0] < xFLeft:
                    xFLeft = intersectxF[0]
                    xFLeftLine = [[x1,y1],[x2,y2]]
                if intersectxF[0] > xFRight:
                    xFRight = intersectxF[0]
                    xFRightLine = [[x1,y1],[x2,y2]]
            if intersectyF is not None:
                if intersectyF[1] < yFTop:
                    yFTop = intersectyF[1]
                    yFTopLine = [[x1,y1],[x2,y2]]
                if intersectyF[1] > yFBottom:
                    yFBottom = intersectyF[1]
                    yFBottomLine = [[x1,y1],[x2,y2]]
            
    # cv2.imwrite("hough_lines_orignal.png", image)

    # Top line has margin of error that effects all courtmapped outputs 
    yOTopLine[0][1] = yOTopLine[0][1]+4
    yOTopLine[1][1] = yOTopLine[1][1]+4

    yFTopLine[0][1] = yFTopLine[0][1]+4
    yFTopLine[1][1] = yFTopLine[1][1]+4


    # Find four corners of the court and display it
    topLeftP = MO.findIntersection(xOLeftLine, yOTopLine, -extraLen, 0, width+extraLen, height)
    topRightP = MO.findIntersection(xORightLine, yFTopLine, -extraLen, 0, width+extraLen, height)
    bottomLeftP = MO.findIntersection(xFLeftLine, yOBottomLine, -extraLen, 0, width+extraLen, height)
    bottomRightP = MO.findIntersection(xFRightLine, yFBottomLine, -extraLen, 0, width+extraLen, height)

    print("Time taken for detection : ", time.time() - start_time)

    

    return {'topLeft': topLeftP, 'topRight': topRightP, 'bottomLeft': bottomLeftP, 'bottomRight': bottomRightP}

# requires dictionary to be in the format of the output of generate_corner_points
def refine_corner_points(dictionary, kernel_size, image, threshold): 

    start_time = time.time()

    image_copy = np.copy(image)

    # image_width, image_height = image.shape[:2]

    def get_nearest_pixel(coordinates, image, threshold): 
        x_axis = coordinates[0]
        y_axis = coordinates[1]

        x1 = x_axis-kernel_size//2
        y1 = y_axis-kernel_size//2
        x2 = x_axis+kernel_size//2
        y2 = y_axis+kernel_size//2

        distances = []

        for x in range(x1, x2+1):
            for y in range(y1, y2+1):
                if image[y][x][0] > threshold and image[y][x][1] > threshold and image[y][x][2] > threshold:
                    distances.append(((x, y), np.sqrt((x-x_axis)**2 + (y-y_axis)**2)))

        sorted_distances = sorted(distances, key=lambda x: x[1])

        # print("sorted_distances = ", sorted_distances)

        return sorted_distances[0][0]
    
    newTopLeft = get_nearest_pixel(dictionary['topLeft'], image, threshold)
    newTopRight = get_nearest_pixel(dictionary['topRight'], image, threshold)
    newBottomLeft = get_nearest_pixel(dictionary['bottomLeft'], image, threshold)
    newBottomRight = get_nearest_pixel(dictionary['bottomRight'], image, threshold)

    # cv2.line(image_copy, newTopLeft, newBottomLeft, (0, 255, 0), 2)
    # cv2.line(image_copy, newTopLeft, newTopRight, (0, 255, 0), 2)
    # cv2.line(image_copy, newBottomLeft, newBottomRight, (0, 255, 0), 2)
    # cv2.line(image_copy, newBottomRight, newTopRight, (0, 255, 0), 2)

    print("time taken to refine points : ", time.time() - start_time)

    return {'topLeft': newTopLeft, 'topRight': newTopRight, 'bottomLeft': newBottomLeft, 'bottomRight': newBottomRight}

def get_line_coordinates(dictionary, image, thickness, color):

    start_time = time.time()

    image_copy = np.zeros_like(image)

    print("time taken to create a copy image : ", time.time() - start_time)

    cv2.line(image_copy, dictionary['topLeft'], dictionary['bottomLeft'], color, thickness)
    cv2.line(image_copy, dictionary['topLeft'], dictionary['topRight'], color, thickness)
    cv2.line(image_copy, dictionary['bottomLeft'], dictionary['bottomRight'], color, thickness)
    cv2.line(image_copy, dictionary['bottomRight'], dictionary['topRight'], color, thickness)

    print("time taken to create lines : ", time.time() - start_time)

    coordinates = []

    # for i in range(len(image_copy)):
    #     for j in range(len(image_copy[0])):
    #         if image_copy[i][j][1] != 0:
    #             coordinates.append((i, j))

    non_zero_green_pixels = np.where(image_copy[:, :, 2] != 0)

    coordinates = list(zip(non_zero_green_pixels[0], non_zero_green_pixels[1]))

    print("time taken to get line coordinates : ", time.time() - start_time)

    return coordinates