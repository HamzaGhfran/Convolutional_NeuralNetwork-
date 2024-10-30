import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread("./images/truck.jpg", cv2.IMREAD_COLOR)

print(img)

cv2.imshow("without normalization", img)
img = img/255

print(img)

cv2.imshow("normalization", img)
# gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# #cv2.imshow("gray_scale", gray_image)

# sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

# #cv2.imshow("HorizontalEdges", sobel_x)
# #cv2.imshow("VerticalEdges", sobel_y)

# magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

# treshold = 50

# edges = magnitude > treshold

# image = Image.fromarray(edges)

# image.show()
cv2.waitKey(5000)
cv2.destroyAllWindows()

