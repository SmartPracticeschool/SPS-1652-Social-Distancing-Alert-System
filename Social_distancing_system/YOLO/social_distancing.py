# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:03:17 2020

@author: asus
"""

#loading camera/ video
#load YOLO, Weights, Labels
#Results Centroid, Bounding Box, Confidence(80/90),Probability(0,1,2)
#Fetches only person values from Object Detection
#Start calculating Euclidean distance
#Violations
#Draw Circles / Rect
#Output

from package import social_distancing_configuration as config
from package.object_detection import detect_people

from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os




labelsPath = os.path.sep.join([config.MODEL_PATH,"coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath= os.path.sep.join([config.MODEL_PATH,"yolov3.weights"])

configPath = os.path.sep.join([config.MODEL_PATH,"yolov3.cfg"])

#COCO 80 CLASSES
print("[INFO] Loading YOLO from disk...")
net=cv2.dnn.readNetFromDarknet(configPath,weightsPath)

if config.USE_GPU:
    
    print("[INFO] setting preferable backend and target to CUDA... ")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
ln=net.getLayerNames()
ln=[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")


vs=cv2.VideoCapture(r"background.mp4"if"background.mp4"else 0)
global writer
writer=None

while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream 
    if not grabbed:
        
        break
    # resize the frame and then detect people (and only people) in ii 
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, 
        personIdx=LABELS.index("person"))
    # initialize the set of indexes that violate the minimum social
    # distance
    violate = set()
    
    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="eucLidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < config.MIN_DIST:
                    violate.add(i) 
                    violate.add(j)
    
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid 
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2) 
        cv2.circle(frame, (cX, cY), 5, color, 1)
    
    text = "SociaL Distancing VioLations: {}".format(len(violate)) 
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
      cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key== ord("q"):
        break
    
   
    if r"sociaL-distance-detector" != "" and writer is None: 
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(r"output.mp4", fourcc, 25, 
                                 (frame.shape[1], frame.shape[0]), True)
    
    if writer is not None:
        writer.write(frame)


cv2.destroyAllWindows()



     



    




