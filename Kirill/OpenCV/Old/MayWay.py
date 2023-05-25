import cv2 as cv
from PIL import Image
import numpy as np

#path = '35Original.png'
path = 'inverted_combined_mask.png'
opened = Image.open(path)

src = cv.imread(cv.samples.findFile(path), cv.IMREAD_GRAYSCALE)

dst = cv.Canny(src, 50, 200, None, 3)
cv.imwrite('canny.png', dst)
canny = Image.open('canny.png')

linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

cdstP = np.zeros(src.shape,dtype=np.uint8)
cdstP.fill(255) 
cdstP = cv.cvtColor(cdstP, cv.COLOR_GRAY2BGR)

if linesP is not None:
  for i in range(0, len(linesP)):
    l = linesP[i][0]
    cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

cv.imwrite('lines.png', cdstP)
lines = Image.open('lines.png')

w_scaled = 400
h_scaled = int(opened.size[1] * w_scaled /opened.size[0])
new_image = Image.new('RGB',(3*w_scaled, h_scaled), (250,250,250))

new_image.paste(opened.resize((w_scaled, h_scaled)),(0,0))
new_image.paste(canny.resize((w_scaled, h_scaled)),(w_scaled,0))
new_image.paste(lines.resize((w_scaled, h_scaled)),(2*w_scaled,0))

new_image
