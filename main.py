import cv2
import torch
import ultralytics
import numpy as np
from torch import cpu
from ultralytics import YOLO


def detect_card():
    model = YOLO("pokemon_card_detector_12_8_2024.pt")

    model.predict(source= "0", show=True, save=False, conf=0.2, line_width=2, save_crop = False, save_txt = False, show_labels = True, show_conf = True, classes=[0])


def train():
    # Create a new YOLO model from scratch
    model = YOLO("yolo11n.yaml")

    model.train(data="dataset_custom.yaml", imgsz=640, batch=120, epochs=200, workers=1, device=0)


if __name__ == "__main__":
    # train()
    detect_card()