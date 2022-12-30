import argparse

import cv2
import torch
import bbox_visualizer as bbv


CLASS_COLOR = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255),
               6: (128, 128, 128), 7: (128, 0, 128)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default='yolov5_ws/yolov5/runs/train/YOLO5-ORIGINAL-ADAMW/weights/best.pt',
                        help="Weights to use for inference")
    parser.add_argument("--target", type=str, help="Target image to do inference on",
                        default="data/raw/images/test/P016_tissue1_144.jpg")
    parser.add_argument("--save-txt", action='store_true', help="Set to save results to .txt file", default=True)
    return parser.parse_args()


def main(args):
    model = torch.hub.load('yolov5_ws/yolov5', 'custom', source='local', path=args.weights, force_reload=True)
    x = cv2.cvtColor(cv2.imread(args.target), cv2.COLOR_BGR2RGB)
    y = model(x)
    if args.save_txt:
        with open("result.txt", "w") as f:
            pred = y.xywhn[0].tolist()
            for p in pred:
                f.write(" ".join([str(i) for i in p]))
                f.write("\n")

    results_xyxy = y.xyxy[0].tolist()
    labels = []
    boxes = []
    classes = []
    for res in results_xyxy:
        labels.append(y.names[int(res[-1])])
        classes.append(int(res[-1]))
        boxes.append([int(a) for a in res[0:4]])

    for box, lbl, cls in zip(boxes, labels, classes):
        color = CLASS_COLOR[cls]
        x = bbv.draw_multiple_rectangles(x, [box], bbox_color=color)
        x = bbv.add_multiple_labels(x, [lbl], [box], text_bg_color=color)

    cv2.imshow('Frame', x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
