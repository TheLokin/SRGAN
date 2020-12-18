import os
import torch
import argparse
import torchvision.utils as utils
import torchvision.transforms as transforms

from PIL import Image
from models import Generator


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution Test.")
parser.add_argument("--image", type=str, metavar="N",
                    help="Image to apply super-resolution.")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--crop-size", type=int, default=400, metavar="N",
                    help="Crop size for the training images (default: 400).")
opt = parser.parse_args()

# Create the necessary folders
if not os.path.exists("test"):
    os.makedirs("test")

# Selection of appropriate treatment equipment
if not torch.cuda.is_available():
    device = "cpu"
else:
    device = "cuda:0"

# Construct SRGAN model
model = Generator(16, opt.upscale_factor).to(device)
checkpoint = torch.load(os.path.join(
    "weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"), map_location=device)
model.load_state_dict(checkpoint["model"])

# Set model eval mode
model.eval()

lr = Image.open(opt.image)
opt.crop_size -= opt.crop_size % opt.upscale_factor
lr = transforms.Resize(opt.crop_size // opt.upscale_factor,
                       interpolation=Image.BICUBIC)(lr)
lr = transforms.ToTensor()(lr)
lr = lr.unsqueeze(0)
lr = lr.to(device)

with torch.no_grad():
    sr = model(lr)

utils.save_image(sr, os.path.join("test", "sr.bmp"))
