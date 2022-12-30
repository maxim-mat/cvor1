import numpy as np
import pandas as pd
import cv2 as cv
import bbox_visualizer as bbv
import matplotlib.pyplot as plt
from PIL import __version__ as PILLOW_VERSION
from torchvision import datasets
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import bbox_visualizer as bbv
import shutil
import torch
import csv

# YOLOv5 PyTorch HUB Inference (DetectionModels only)

model_name='yolov5_ws/yolov5/runs/train/exp5/weights/best.pt'
model = torch.hub.load('yolov5_ws/yolov5', 'custom', source='local', path = model_name, force_reload = True)


K = 45

videoPath = 'data/raw/videos/P022_balloon1.wmv'
outName = videoPath.split("/")[-1]
outName = outName.split(".")[0]
right_segments = outName + "_right"
left_segments = outName + "_left"


def findMostViews(views, catagories):
    counts = []
    for  c in catagories:
        counts.append(views.count(c))
    return catagories[np.argmax(counts)]


def classToTool(class_):
    tool = int(class_/2)
    if tool == 0:
        tool = 3
    elif tool == 1:
        tool = 1
    elif tool == 2:
        tool = 2
    else:
        tool = 0
    return "T" + str(tool)


def createSegments(predictions):
    segments = []
    start_p = 0
    current = predictions[0]
    for pre in range(0,len(predictions)):
        if predictions[pre] == current:
            continue
        else:
            segments.append([start_p, pre - 1, classToTool(current)])
            current = predictions[pre]
            start_p = pre
    if start_p < len(predictions):
        segments.append([start_p, len(predictions), classToTool(current)])
    return segments


## https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/      ## basic opencv tutorial
## https://github.com/shoumikchow/bbox-visualizer  ## bbox_visualizer git with examples


cap = cv.VideoCapture(videoPath)

right_labels = []
left_labels = []

right_k_labels = []
left_k_labels = []
for j in range(0,K):
    right_k_labels.append(6)
    left_k_labels.append(7)


# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(outName + '.mp4',fourcc, 30.0, (640,480))

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
i=0
while (cap.isOpened()):
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = model(frame)
        #print(results.xyxy)

        # add bounding boxes
        # bbox = [xmin, ymin, xmax, ymax]
        results_xyxy = results.xyxy[0].tolist()
        labels = []
        boxes = []
        for res in results_xyxy:
            labels.append(results.names[int(res[5])])
            boxes.append([int(a) for a in res[0:4]])
            if int(res[5])%2 == 0:
                right_k_labels.append(int(res[5]))
                del right_k_labels[0]
                right_labels.append(findMostViews(right_k_labels, list(results.names.keys())))
            else:
                left_k_labels.append(int(res[5]))
                del left_k_labels[0]
                left_labels.append(findMostViews(left_k_labels, list(results.names.keys())))



        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        frame = bbv.draw_multiple_rectangles(frame, boxes,bbox_color=(255,0,0))
        frame = bbv.add_multiple_labels(frame, labels, boxes,text_bg_color=(255,0,0))

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame,results.names[right_labels[-1]] +" " + results.names[left_labels[-1]],(50,50), font, 1, (0,255,255), 2, cv.LINE_4)

        # Display the resulting frame
        cv.imshow('Frame', frame)

        # write the frame
        out.write(frame)



        # Press Q on keyboard to  exit
        if cv.waitKey(33) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv.destroyAllWindows()

# write the segments

with open(right_segments+".txt","w", newline="") as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(createSegments(right_labels))

with open(left_segments+".txt","w", newline="") as f:
    writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    writer.writerows(createSegments(left_labels))


predictedSegmentsPath = ""
trueSegmentsPath = ""

def convertSegmentsToList(segments):
    segList = []
    for segment in segments:
        for i in range(segment[0],segment[1]+1):
            segList.append(segment[2])
    return segList


def readSegmentFile(path):
    with open(path,"r") as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        return reader.readLines()

def compareToolUsage(predictions, groundTruth):
    return

predictedSegments = readSegmentFile(predictedSegmentsPath)
trueSegments = readSegmentFile(trueSegmentsPath)
predictedSegments = convertSegmentsToList(predictedSegments)
trueSegments = convertSegmentsToList(trueSegments)
compareToolUsage(predictedSegments, trueSegments)
