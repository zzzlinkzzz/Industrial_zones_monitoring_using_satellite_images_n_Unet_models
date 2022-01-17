import torch
import torch.nn as nn
from Unet_model import Unet_model

from PIL import Image

img_dir = './data/val/images/22-76.png'
trained_model = './cp_unet/CP15.pth'
img = Image.open(img_dir)
model = torch.load(trained_model)