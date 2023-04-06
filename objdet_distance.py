import cv2
import numpy as np
import random
import time
import pyttsx3
import threading
import argparse
from math import pow, sqrt
from imutils.video import FPS

video=cv2.VideoCapture(0)

R=random.randint(0, 255)
G=random.randint(0, 255)
B=random.randint(0, 255)

# CLASSES = ["Background", "Aeroplane", "Bicycle", "Bird", "Boat",
# 	"Bottle", "Bus", "Car", "Cat", "Chair", "Cow", "Diningtable",
# 	"Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep",
# 	"Sofa", "Train", "TVmonitor"]

#color=[(R,G,B) for i in CLASSES]
parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video', type = str, default = 'demo.mp4', help = 'Video file path. If no path is given, video is captured using device.')

parser.add_argument('-m', '--model', default = 'SSD_MobileNet.caffemodel', help = "Path to the pretrained model.")
    
parser.add_argument('-p', '--prototxt', default = 'SSD_MobileNet_prototxt.txt', help = 'Prototxt of the model.')

parser.add_argument('-l', '--labels', default = 'class_labels.txt', help = 'Labels of the dataset.')

parser.add_argument('-c', '--confidence', type = float, default = 0.8, help='Set confidence for detecting objects')

args = parser.parse_args(args=[])


labels  = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]


# Generate random bounding_box_color for each label
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))


# Load model
print("\nLoading model...\n")

net=cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()
frame_no = 0

while cap.isOpened():

    frame_no = frame_no+1

    # Capture one frame after another
    ret, frame = cap.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

engine = pyttsx3.init()

while True:
    
    ret, frame=video.read()
    frame = cv2.resize(frame,(640,480))
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)
    net.setInput(blob)
    detections=net.forward()
    #print(detection)
    for i in np.arange(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>0.8:
            id1=detections[0,0,i,1]
            def sayItem() :
             voices = engine.getProperty('voices')
             engine.setProperty('voice',voices[1].id)
             #engine.setProperty('rate',10)
             #engine.say("Be Careful"+labels[int (id1)]+"Ahead");
             a=labels[int (id1)]
             engine.setProperty('volume',0.9)
             #if(True):
             if(a==labels[int (id1)]):
               engine.runAndWait() 
             else:
               engine.runAndWait()  
                   
            x=threading.Thread(target=sayItem, daemon=True)
            x.start()
           
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY)=box.astype("int")
            cv2.rectangle(frame, (startX -1,startY-40), (endX+1,startY-2),COLORS[int(id1)],-1)
            cv2.rectangle(frame, (startX,startY),(endX,endY),COLORS[int(id1)], 2)
            cv2.putText(frame, labels[int (id1)], (startX+10,startY-15),cv2.FONT_HERSHEY_SIMPLEX,(0.7),(255,255,255))
    pos_dict = dict()
    coordinates = dict()

    # Focal length (in millimeters)
    F = 50

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args.confidence:

            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            coordinates[i] = (startX, startY, endX, endY)

            # Mid point of bounding box
            x_mid = round((startX+endX)/2,4)
            y_mid = round((startY+endY)/2,4)

            height = round(endY-startY,4)

            # Distance from camera based on triangle similarity
            dis = (165 * F)/height
            distance=round(dis,2)
            print("Distance(cms):{dist}\n".format(dist=distance))
            engine.say(labels[int (id1)]+"is at distance of"+"{dist}\n".format(dist=distance)+"centimeters");

            # Mid-point of bounding boxes (in cm) based on triangle similarity technique
            x_mid_cm = (x_mid * distance) / F
            y_mid_cm = (y_mid * distance) / F
            pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

    # Distance between every object detected in a frame
    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check if distance less than 1 foot (300 mm approx):
                if dist < 300:
                    close_objects.add(i)
                    close_objects.add(j)

    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = (0,0,255)
        else:
            COLOR = (0,255,0)
     
        (startX, startY, endX, endY) = coordinates[i]

        cv2.rectangle(frame,(startX,startY), (endX, endY), COLOR, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        # Convert mms to feet
        cv2.putText(frame, 'Distance: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)     
    cv2.imshow("Frame",frame)
    cv2.resizeWindow('Frame',1400,1200)
    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()
# stop the timer and display FPS information
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
#print("FPS: {:.2f}".format(fps.fps()))
# Clean
cap.release()
cv2.destroyAllWindows() 
   