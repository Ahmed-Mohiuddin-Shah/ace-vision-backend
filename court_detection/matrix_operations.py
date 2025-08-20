
def determinant(a, b):
    return a[0] * b[1] - a[1] * b[0]
    
def findIntersection(line1, line2, xStart, yStart, xEnd, yEnd):

    xDiff = (line1[0][0]-line1[1][0],line2[0][0]-line2[1][0])
    yDiff = (line1[0][1]-line1[1][1],line2[0][1]-line2[1][1])
    div = determinant(xDiff, yDiff)
    if div == 0:
        return None
    
    d = (determinant(*line1), determinant(*line2))
    x = int(determinant(d, xDiff) / div)
    y = int(determinant(d, yDiff) / div)
    
    if (x<xStart) or (x>xEnd):
        return None
    if (y<yStart) or (y>yEnd):
        return None
    return x,y
