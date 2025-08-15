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

torch.cuda.empty_cache()

NumCell = 104  # number of cells
NumClass = 11  # number of classes except background class
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
model.load_state_dict(torch.load("YOLIC_M2.pth.tar", map_location=torch.device('cpu')))



# Use openvino to convert the model
import openvino as ov

output_dir = r"ai_models/openvino_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Convert the PyTorch model to OpenVINO format
import torch
dummy_input = torch.randn(1, 3, 224, 224)
ov_model = ov.convert_model(model, example_input=dummy_input)
# Save the OpenVINO model
ov.save_model(ov_model, os.path.join(output_dir, "yolic_mobilenet_v2.xml"))