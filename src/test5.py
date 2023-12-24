import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a color image in grayscale
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

cv2.imwrite("grayscale.png", img)
#cv2.waitKey(0)

edges = cv2.Canny(img, 10, 200)

cv2.imwrite("canny.png", edges)
#cv2.waitKey(0)

#ret, edges = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)


#cv2.imshow("canny", edges)
#cv2.waitKey(0)