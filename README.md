# MY_PG_INTERNSHIP
Using Computer Vision Techniques lanes on roads got identified.

This project implements a classical computer vision–based lane detection pipeline for both images and videos. The core logic is handled in lane_detection.py, which applies operations such as color filtering, edge detection (Canny), region of interest masking, and Hough line transformation to detect lane markings. The detected lanes are then optionally smoothed across frames using a buffer to provide temporal consistency in video streams. Unlike deep learning approaches, this implementation relies entirely on traditional image processing techniques, making it lightweight.

The main.py script acts as the entry point of the project. It first processes all test images from the data/test_images/ directory, saving lane-marked outputs to out/images/. It then processes videos from the data/test_videos/ directory, applying the same pipeline frame by frame. A deque buffer enables temporal smoothing for stable lane visualization in video output. This makes the project a complete end-to-end pipeline for evaluating lane detection on both static images and driving footage.
