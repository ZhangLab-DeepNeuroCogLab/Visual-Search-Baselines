import sys
import cv2
import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle
import random
import copy

import os
import shutil
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

class NaturalDesign(Dataset): 
  def __init__(self, dataset_dir): 
    self.dataset_dir = dataset_dir
    self.context_dir = os.path.join(self.dataset_dir, "stimuli")
    self.bbox_dir = os.path.join(self.dataset_dir, "gt")

  def __len__(self):
    return len(os.listdir(self.context_dir))
  
  def __getitem__(self, idx):
    context_file = os.path.join(self.context_dir, 'img{:03}.jpg'.format(idx))
    bbox_file = os.path.join(self.bbox_dir, 'gt{:d}.jpg'.format(idx))
    
    # process context img
    img = cv2.imread(context_file)
    
    # calculate the bounding box cordinates througn gt img
    gt_img = cv2.imread(bbox_file)
    img_width, img_height = gt_img.shape[1], gt_img.shape[0]
    gt_img = gt_img.transpose(2, 0, 1)
    r_bbox, c_bbox = np.where(gt_img[0] == np.max(gt_img[0]))
    xmin, ymin, w, h = c_bbox.min(), r_bbox.min(), c_bbox.max()-c_bbox.min(), r_bbox.max()-r_bbox.min()
    bbox_relative = torch.tensor([xmin / img_width, ymin / img_height, w / img_width, h / img_height])

    return img, bbox_relative