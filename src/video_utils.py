"""
Helper functions for video I/O.
"""

import cv2

def read_video(video_path):
    return cv2.VideoCapture(video_path)

def write_video(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
