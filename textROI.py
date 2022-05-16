import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
min_confidence=0.5

east_decorator='frozen_east_text_detection.pb'

frame_size=320
padding=0.05

def textROI(image):
    # load the input image and grab the image dimensions
    orig=image.copy()
    (origH,origW) = image.shape[:2]
    
    # 자동차 이미지를 잘라오게 되면 자동차의 크기에 따라 이미지의 사이즈가 달라지기 때문에
    # (정사각형이 아니기 때문에) 번호판 글씨가 왜곡(늘려지거나 줄여지는 현상)될 수 있다(번호판은 차의 중앙에 있다)
    # 그러므로 늘리거나 줄임없이 정사각형 이미지로 (320x320) 잘라내기 위해 다음 작업을 실행한다.
    rW=origW / float(frame_size)
    rH=origH/ float(frame_size)
    newW=int(origW/rH)
    center=int(newW / 2)
    start=center - int (frame_size/2)
    
    image=cv2.resize(image,(newW,frame_size))
    scale_image=image[0:frame_size,start:start+frame_size]
    (H,W) = scale_image.shape[:2]
    
    cv2.imshow("orig", orig)
    cv2.imshow("resize", image)
    cv2.imshow("scale_image", scale_image)
    
    # define the two output layer names for the EAST detector model 
    layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(east_decorator)

    # construct a blob from the image 
    blob = cv2.dnn.blobFromImage(image, 1.0, (frame_size, frame_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities)
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):

                if scoresData[x] < min_confidence:
                        continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
    
    # apply non-maxima suppression 
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            dX = int((endX - startX) * padding)
            dY = int((endY - startY) * padding)

            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))

            # extract the actual padded ROI
            return ([startX, startY, endX, endY], orig[startY:endY, startX:endX])