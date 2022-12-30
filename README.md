CVOR1
==============================

This is a repo for hw1 in cvor course

Contents:

	data - raw and processed data
	predict.py - object detection on single image
	video.py - object detection on video
	report, report_pdf - notebook and html files for report
	requirements.txt - required packages to run
	Segmented_Videos.zip - results of object detection on all videos + videos with detected objects
	yolov5_ws - workspace for yolo5 implementation we used
	yolov5_ws/yolov5/runs - contains all tested variations and model parameters
	
Usage:
	
	pip install -r requirements.txt
	python video.py --target <your video>
	python predict.py --target <your image> [--save-txt] 
		save-txt flag will write object detection to result.txt
