import os
import cv2
import lpips
import torch
import argparse
import torchvision.utils as utils

from tqdm import tqdm
from models import Generator
from dataset import DatasetFromFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sewar.full_ref import mse, rmse, psnr, ssim, msssim


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution Comparation.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
opt = parser.parse_args()

# Create the necessary folders
if not os.path.exists(os.path.join("test", "SRGAN", "2x2x")):
    os.makedirs(os.path.join("test", "SRGAN", "2x2x"))

# Selection of appropriate treatment equipment
if not torch.cuda.is_available():
    device = "cpu"
else:
    device = "cuda:0"

# Load dataset
dataset = DatasetFromFolder(opt.dataset, 800, 4)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct SRGAN model
model = Generator(16, 2).to(device)
checkpoint = torch.load(os.path.join(
    "weight", "SRGAN", "netG_2x.pth"), map_location=device)
model.load_state_dict(checkpoint["model"])

# Set model eval mode
model.eval()

# Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
lpips_loss = lpips.LPIPS(net="vgg").to(device)

# Evaluate algorithm performance
total_mse_value = 0
total_rmse_value = 0
total_psnr_value = 0
total_ssim_value = 0
total_ms_ssim_value = 0
total_lpips_value = 0

# Start evaluate model performance
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (input, target) in progress_bar:
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(model(lr))

    utils.save_image(lr, os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_lr.bmp"))
    utils.save_image(hr, os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_hr.bmp"))
    utils.save_image(sr, os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_sr.bmp"))

    lr_img = cv2.imread(os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_lr.bmp"))
    dst_img = cv2.imread(os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_hr.bmp"))
    src_img = cv2.imread(os.path.join(
        "test", "2x2x", "SRGAN_" + str(i + 1) + "_sr.bmp"))

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value += mse_value
    total_rmse_value += rmse_value
    total_psnr_value += psnr_value
    total_ssim_value += ssim_value[0]
    total_ms_ssim_value += ms_ssim_value.real
    total_lpips_value += lpips_value.item()

avg_mse_value = total_mse_value / len(dataloader)
avg_rmse_value = total_rmse_value / len(dataloader)
avg_psnr_value = total_psnr_value / len(dataloader)
avg_ssim_value = total_ssim_value / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value / len(dataloader)
avg_lpips_value = total_lpips_value / len(dataloader)

print("\n=== Performance summary (upsampling x2x2 vs upsampling x4)\n" +
      "Avg MSE: {:.4f}\n".format(avg_mse_value) +
      "Avg RMSE: {:.4f}\n".format(avg_rmse_value) +
      "Avg PSNR: {:.4f}\n".format(avg_psnr_value) +
      "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
      "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))
