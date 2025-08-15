import itertools

import cv2
from torchvision.models import mobilenet_v2
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import random
from PIL import Image
import argparse
import numpy as np
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os.path
import os

from ultralytics import YOLO

torch.cuda.empty_cache()

# Load yolov8.pt model

model_path = "yolov8.pt"  # replace with your model path
model = YOLO(model_path)  # Assuming YOLO is defined elsewhere


# Use openvino to convert the model
import openvino as ov

output_dir = r"ai_models/openvino_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert the PyTorch model to OpenVINO format
import torch
dummy_input = torch.randn
ov_model = ov.convert_model(model, example_input=dummy_input)
# Save the OpenVINO model
ov.save_model(ov_model, os.path.join(output_dir, "yolov8.xml"))