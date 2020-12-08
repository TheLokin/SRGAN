import os
import argparse

import cv2
import lpips
import torch
import torchvision.utils as utils

from sewar.full_ref import mse, rmse, psnr, ssim, msssim

from PIL import Image
from models import Generator
from torchvision.transforms import ToTensor


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution.")
parser.add_argument("--lr", type=str, required=True,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str, required=True,
                    help="Raw high resolution image name.")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default: 4).")
parser.add_argument("--model-path", default="weight/SRGAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: 'weight/SRGAN_4x.pth').")
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
model.load_state_dict(torch.load(opt.model_path, map_location=device))

# Set model eval mode
model.eval()

# Reference sources from 'https://github.com/richzhang/PerceptualSimilarity'
lpips_loss = lpips.LPIPS(net="vgg").to(device)

# Load image
lr = Image.open(opt.lr)
hr = Image.open(opt.hr)
lr = ToTensor()(lr).unsqueeze(0)
hr = ToTensor()(hr).unsqueeze(0)
lr = lr.to(device)
hr = hr.to(device)

with torch.no_grad():
    sr = model(lr)

utils.save_image(lr, "test/lr.bmp")
utils.save_image(hr, "test/hr.bmp")
utils.save_image(sr, "test/sr.bmp")

# Evaluate performance
src_img = cv2.imread("test/sr.bmp")
dst_img = cv2.imread("test/hr.bmp")

mse_value = mse(src_img, dst_img)
rmse_value = rmse(src_img, dst_img)
psnr_value = psnr(src_img, dst_img)
ssim_value = ssim(src_img, dst_img)
ms_ssim_value = msssim(src_img, dst_img)
lpips_value = lpips_loss(sr, hr)

print("====================== Performance summary ======================")
print(f"MSE: {mse_value:.2f}\n"
      f"RMSE: {rmse_value:.2f}\n"
      f"PSNR: {psnr_value:.2f}\n"
      f"SSIM: {ssim_value[0]:.4f}\n"
      f"MS-SSIM: {ms_ssim_value.real:.4f}\n"
      f"LPIPS: {lpips_value.item():.4f}")
print("============================== End ==============================")
