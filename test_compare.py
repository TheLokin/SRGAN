import os

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


# Create the necessary folders
if not os.path.exists("test"):
    os.makedirs("test")

# Selection of appropriate treatment equipment
if not torch.cuda.is_available():
    device = "cpu"
else:
    device = "cuda:0"

# Load dataset
dataset = DatasetFromFolder(f"data/4x/test/input", f"data/4x/test/target")
dataloader = DataLoader(dataset, pin_memory=True)

# Construct SRGAN model
model = Generator(16, 2).to(device)
model.load_state_dict(torch.load("SRGAN_2x.pth", map_location=device))

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
    # Set model gradients to zero
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(lr)
        sr = model(sr)

    utils.save_image(lr, f"test/lr.bmp")
    utils.save_image(sr, f"test/sr.bmp")
    utils.save_image(hr, f"test/hr.bmp")

    # Evaluate performance
    src_img = cv2.imread(f"test/sr.bmp")
    dst_img = cv2.imread(f"test/hr.bmp")

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

print("\n")
print("====================== Performance summary ======================")
print(f"Avg MSE: {avg_mse_value:.2f}\n"
      f"Avg RMSE: {avg_rmse_value:.2f}\n"
      f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n"
      f"Avg MS-SSIM: {avg_ms_ssim_value:.4f}\n"
      f"Avg LPIPS: {avg_lpips_value:.4f}")
print("============================== End ==============================")
