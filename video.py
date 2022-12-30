import argparse
import numpy as np
import cv2 as cv
import bbox_visualizer as bbv
import torch
import csv


CLASS_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255),
               6: (128, 128, 128), 7: (128, 0, 128)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default='yolov5_ws/yolov5/runs/train/YOLO5-ORIGINAL-ADAMW/weights/best.pt',
                        help="Weights to use for inference")
    parser.add_argument("--target", type=str, help="Target video to do inference on",
                        default="data/raw/Videos/P022_balloon1.wmv")
    return parser.parse_args()


def findMostViews(views, catagories):
    counts = []
    for c in catagories:
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
    for pre in range(0, len(predictions)):
        if predictions[pre] == current:
            continue
        else:
            segments.append([start_p, pre - 1, classToTool(current)])
            current = predictions[pre]
            start_p = pre
    if start_p < len(predictions):
        segments.append([start_p, len(predictions), classToTool(current)])
    return segments


def main(args):

    model = torch.hub.load('yolov5_ws/yolov5', 'custom', source='local', path=args.weights, force_reload=True)
    K = 45
    videoPath = args.target
    outName = videoPath.split("/")[-1]
    outName = outName.split(".")[0]
    right_segments = outName + "_right"
    left_segments = outName + "_left"

    cap = cv.VideoCapture(videoPath)

    right_labels = []
    left_labels = []

    right_k_labels = []
    left_k_labels = []
    for j in range(0, K):
        right_k_labels.append(6)
        left_k_labels.append(7)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(outName + '.mp4', fourcc, 30.0, (640, 480))

    if not cap.isOpened():
        print("Error opening video stream or file")

    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = model(frame)
            results_xyxy = results.xyxy[0].tolist()
            labels = []
            boxes = []
            classes = []
            for res in results_xyxy:
                labels.append(results.names[int(res[5])])
                boxes.append([int(a) for a in res[0:4]])
                classes.append(int(res[-1]))
                if int(res[5]) % 2 == 0:
                    right_k_labels.append(int(res[5]))
                    del right_k_labels[0]
                    right_labels.append(findMostViews(right_k_labels, list(results.names.keys())))
                else:
                    left_k_labels.append(int(res[5]))
                    del left_k_labels[0]
                    left_labels.append(findMostViews(left_k_labels, list(results.names.keys())))

            for box, lbl, cls in zip(boxes, labels, classes):
                color = CLASS_COLOR[cls]
                frame = bbv.draw_multiple_rectangles(frame, [box], bbox_color=color)
                frame = bbv.add_multiple_labels(frame, [lbl], [box], text_bg_color=color)

            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame, results.names[right_labels[-1]] + " " + results.names[left_labels[-1]], (50, 50), font, 1,
                       (0, 255, 255), 2, cv.LINE_4)

            cv.imshow('Frame', frame)
            out.write(frame)

            if cv.waitKey(33) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    out.release()

    cv.destroyAllWindows()

    with open(right_segments+".txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(createSegments(right_labels))

    with open(left_segments+".txt", "w", newline="") as f:
        writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(createSegments(left_labels))


if __name__ == "__main__":
    args = parse_args()
    main(args)
