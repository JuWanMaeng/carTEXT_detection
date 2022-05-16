import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

from carROI import carROI
from textROI import textROI
from textREAD import textRead

file_name='image/parking_01.jpg'

# Loading image
img=cv2.imread(file_name)
img_copy=img.copy()
([x,y,w,h],car_image) = carROI(img)
([startX,startY,endX,endY], text_image) = textROI(car_image)

#process_image = processROI(text_image)

text = textRead(text_image)

cv2.rectangle(img_copy, (x+startX, y+startY), (x+endX, y+endY), (0, 255, 0), 2)

cv2.putText(img_copy, text, (x+startX, y+startY-10),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# show the output image
cv2.imshow("OCR Text Recognition : "+text, img_copy)
cv2.imshow('plate_img',text_image)

cv2.waitKey(0)
cv2.destroyAllWindows()