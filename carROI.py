import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import pytesseract

min_confidence=0.5
fil_name='image/image_03.png'

east_decorator='frozen_east_text_detection.pb'

fram_size=320
padding=0.05

# load yolo
net=cv2.dnn.readNet('yolo/yolov3.weights','yolo/yolov3.cfg')
layer_names=net.getLayerNames()
output_layers=[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def carROI(image):
    height,width,channels=image.shape
    # detecting objects
    blob=cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True,crop=False)
    
    net.setInput(blob)
    outs=net.forward(output_layers)
    
    # showing informations on the screen
    confidences=[]
    boxes=[]
    img_cars=[]
    
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            
            # filter only 'car'
            if class_id==2 and confidence > min_confidence:
                # object detected
                center_x=int(detection[0] * width)
                center_y=int(detection[1] * height)
                w=int(detection[2] * width)
                h=int(detection[3] * height)
                
                # rectangle coordinates
                x=int(center_x - w / 2)
                y=int(center_y - h / 2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,min_confidence,0.4) # 제일 좋은 box의 index를 리턴
    
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            img_cars.append(image[y:y+h,x:x+w])
            return (boxes[i],image[y:y+h,x:x+w])
        
        
                
                