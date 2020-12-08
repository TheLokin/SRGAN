import os
import argparse

import cv2
import lpips
import torch
import torchvision.utils as utils

from sewar.full_ref import mse, rmse, psnr, ssim, msssim

from tqdm import tqdm
from PIL import Image
from models import Generator
from dataset import DatasetFromFolder
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution.")
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

# Load dataset
dataset = DatasetFromFolder(
    f"data/{opt.upscale_factor}x/test/input", f"data/{opt.upscale_factor}x/test/target")
dataloader = DataLoader(dataset, pin_memory=True)

# Construct SRGAN model
model = Generator(16, opt.upscale_factor).to(device)
model.load_state_dict(torch.load(opt.model_path, map_location=device))

# Set model eval mode
model.eval()

# Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
lpips_loss = lpips.LPIPS(net="vgg").to(device)

# Evaluate algorithm performance
total_mse_value = [0, 0, 0, 0]
total_rmse_value = [0, 0, 0, 0]
total_psnr_value = [0, 0, 0, 0]
total_ssim_value = [0, 0, 0, 0]
total_ms_ssim_value = [0, 0, 0, 0]
total_lpips_value = [0, 0, 0, 0]

size = 200 * opt.upscale_factor

# Start evaluate model performance
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, (input, target) in progress_bar:
    # Set model gradients to zero
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(lr)

    utils.save_image(lr, f"test/lr.bmp")
    utils.save_image(sr, f"test/sr.bmp")
    utils.save_image(hr, f"test/hr.bmp")

    # Evaluate performance
    lr_img = cv2.imread(f"test/lr.bmp")
    src_img = cv2.imread(f"test/sr.bmp")
    dst_img = cv2.imread(f"test/hr.bmp")

    # Raw high resolution image
    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[0] += mse_value
    total_rmse_value[0] += rmse_value
    total_psnr_value[0] += psnr_value
    total_ssim_value[0] += ssim_value[0]
    total_ms_ssim_value[0] += ms_ssim_value.real
    total_lpips_value[0] += lpips_value.item()

    # Nearest neighbor interpolation
    src_img = cv2.resize(lr_img, (size, size), interpolation=cv2.INTER_NEAREST)
    sr = ToTensor()(src_img).unsqueeze(0)
    sr = hr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[1] += mse_value
    total_rmse_value[1] += rmse_value
    total_psnr_value[1] += psnr_value
    total_ssim_value[1] += ssim_value[0]
    total_ms_ssim_value[1] += ms_ssim_value.real
    total_lpips_value[1] += lpips_value.item()

    # Bilinear interpolation
    src_img = cv2.resize(lr_img, (size, size), interpolation=cv2.INTER_LINEAR)
    sr = ToTensor()(src_img).unsqueeze(0)
    sr = hr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[2] += mse_value
    total_rmse_value[2] += rmse_value
    total_psnr_value[2] += psnr_value
    total_ssim_value[2] += ssim_value[0]
    total_ms_ssim_value[2] += ms_ssim_value.real
    total_lpips_value[2] += lpips_value.item()

    # Bicubic interpolation
    src_img = cv2.resize(lr_img, (size, size), interpolation=cv2.INTER_CUBIC)
    sr = ToTensor()(src_img).unsqueeze(0)
    sr = hr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)
    lpips_value = lpips_loss(sr, hr)

    total_mse_value[3] += mse_value
    total_rmse_value[3] += rmse_value
    total_psnr_value[3] += psnr_value
    total_ssim_value[3] += ssim_value[0]
    total_ms_ssim_value[3] += ms_ssim_value.real
    total_lpips_value[3] += lpips_value.item()

avg_mse_value = total_mse_value[0] / len(dataloader)
avg_rmse_value = total_rmse_value[0] / len(dataloader)
avg_psnr_value = total_psnr_value[0] / len(dataloader)
avg_ssim_value = total_ssim_value[0] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[0] / len(dataloader)
avg_lpips_value = total_lpips_value[0] / len(dataloader)

print("\n")
print("==== Performance summary with raw high resolution image =====")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg LPIPS: {avg_lpips_value:.4f}")

avg_mse_value = total_mse_value[1] / len(dataloader)
avg_rmse_value = total_rmse_value[1] / len(dataloader)
avg_psnr_value = total_psnr_value[1] / len(dataloader)
avg_ssim_value = total_ssim_value[1] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[1] / len(dataloader)
avg_lpips_value = total_lpips_value[1] / len(dataloader)

print("== Performance summary with nearest neighbor interpolation ==")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg LPIPS: {avg_lpips_value:.4f}")

avg_mse_value = total_mse_value[2] / len(dataloader)
avg_rmse_value = total_rmse_value[2] / len(dataloader)
avg_psnr_value = total_psnr_value[2] / len(dataloader)
avg_ssim_value = total_ssim_value[2] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[2] / len(dataloader)
avg_lpips_value = total_lpips_value[2] / len(dataloader)

print("====== Performance summary with bilinear interpolation ======")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg LPIPS: {avg_lpips_value:.4f}")

avg_mse_value = total_mse_value[3] / len(dataloader)
avg_rmse_value = total_rmse_value[3] / len(dataloader)
avg_psnr_value = total_psnr_value[3] / len(dataloader)
avg_ssim_value = total_ssim_value[3] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[3] / len(dataloader)
avg_lpips_value = total_lpips_value[3] / len(dataloader)

print("====== Performance summary with bicubic interpolation =======")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg LPIPS: {avg_lpips_value:.4f}")
print("=========================== End =============================")
