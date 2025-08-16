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
import matplotlib.pyplot as plt
import os

NumCell = 104
NumClass = 11  # number of classes except background class
model = mobilenet_v2()  # load the model
model.classifier[1] = nn.Linear(1280, NumCell * (NumClass + 1))
model.load_state_dict(torch.load("./ai_models/yolic_m2/yolic_m2.pth.tar"))
save_name = 'Outdoor_mobilenet'  # name of the model