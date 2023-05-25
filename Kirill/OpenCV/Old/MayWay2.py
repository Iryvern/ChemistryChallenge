import cv2
import numpy as np
import math, os
import imutils

img = cv2.imread("inverted_combined_mask.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(gray, 200, 100)
edges = cv2.dilate(edges, None, iterations=1)
edges = cv2.erode(edges, None, iterations=1)

cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sorting the contours to find the largest and smallest one
c1 = max(cnts, key=cv2.contourArea)
c2 = min(cnts, key=cv2.contourArea)

# determine the most extreme points along the contours
extLeft1 = tuple(c1[c1[:, :, 0].argmin()][0])
extRight1 = tuple(c1[c1[:, :, 0].argmax()][0])
extLeft2 = tuple(c2[c2[:, :, 0].argmin()][0])
extRight2 = tuple(c2[c2[:, :, 0].argmax()][0])

# show contour
cimg = cv2.drawContours(img, cnts, -1, (0,200,0), 2)

# set y of left point to y of right point
lst1 = list(extLeft1)
lst1[1] = extRight1[1]
extLeft1 = tuple(lst1)

lst2 = list(extLeft2)
lst2[1] = extRight2[1]
extLeft2= tuple(lst2)

# compute the distance between the points (x1, y1) and (x2, y2)
dist1 = math.sqrt( ((extLeft1[0]-extRight1[0])**2)+((extLeft1[1]-extRight1[1])**2) )
dist2 = math.sqrt( ((extLeft2[0]-extRight2[0])**2)+((extLeft2[1]-extRight2[1])**2) )

# draw lines
cv2.line(cimg, extLeft1, extRight1, (255,0,0), 1)
cv2.line(cimg, extLeft2, extRight2, (255,0,0), 1)

# draw the distance text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255,0,0)
lineType = 1
cv2.putText(cimg,str(dist1),(155,100),font, fontScale, fontColor, lineType)
cv2.putText(cimg,str(dist2),(155,280),font, fontScale, fontColor, lineType)

# show image
cv2.imshow("Image", img)
cv2.waitKey(0)
