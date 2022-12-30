import argparse

import cv2
import torch
import os


CLASSES_ENCODING = {'Right_Scissors': 0, 'Left_Scissors': 1, 'Right_Needle_driver': 2, 'Left_Needle_driver': 3,
                    'Right_Forceps': 4, 'Left_Forceps': 5, 'Right_Empty': 6, 'Left_Empty': 7}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str,
                        default='yolov5_ws/yolov5/runs/train/YOLO5-ORIGINAL-ADAMW/weights/best.pt',
                        help="Weights to use for inference")
    parser.add_argument("--target", type=str, help="Target image to do inference on")
    parser.add_argument("--save-txt", action='store_true', help="Set to save results to .txt file")
    return parser.parse_args()


def main(args):
    model_name = args.weights
    model = torch.hub.load('yolov5_ws/yolov5', 'custom', source='local', path=model_name, force_reload=True)
    img_path = args.target
    img_path = "data/raw/images/training/P016_balloon1_9.jpg"
    x = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    y = model(x)
    # if args.save_txt:
    with open("result.txt", "w") as f:
        f.write(y)

    result_xyxy = y.xyxy[0].tolist()


if __name__ == "__main__":
    args = parse_args()
    main(args)
