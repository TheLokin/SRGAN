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
    description="Photo-Realistic Single Image Super-Resolution Benchmark.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--crop-size", type=int, default=400, metavar="N",
                    help="Crop size for the training images (default: 400).")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
opt = parser.parse_args()

# Selection of appropriate treatment equipment
if not torch.cuda.is_available():
    device = "cpu"
else:
    device = "cuda:0"

# Load dataset
dataset = DatasetFromFolder(opt.dataset, opt.crop_size, opt.upscale_factor)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct SRGAN model
model = Generator(16, opt.upscale_factor).to(device)
checkpoint = torch.load(os.path.join(
    "weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"), map_location=device)
model.load_state_dict(checkpoint["model"])

# Set model eval mode
model.eval()

# Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
lpips_loss = lpips.LPIPS(net="vgg").to(device)

# Algorithm performance
total_mse_value = [0, 0, 0, 0]
total_rmse_value = [0, 0, 0, 0]
total_psnr_value = [0, 0, 0, 0]
total_ssim_value = [0, 0, 0, 0]
total_ms_ssim_value = [0, 0, 0, 0]
total_lpips_value = [0, 0, 0, 0]

# Start evaluate model performance
for _, (input, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(lr)

    utils.save_image(lr, "lr.bmp")
    utils.save_image(hr, "hr.bmp")
    utils.save_image(sr, "sr.bmp")

    lr_img = cv2.imread("lr.bmp")
    hr_img = cv2.imread("hr.bmp")

    # Evaluate performance
    for i, (src_img, dst_img) in enumerate([
        (cv2.imread("sr.bmp"), hr_img),
        (cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_NEAREST), hr_img),
        (cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_LINEAR), hr_img),
        (cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_CUBIC), hr_img)
    ]):
        sr = ToTensor()(src_img).unsqueeze(0)
        sr = hr.to(device)

        mse_value = mse(src_img, dst_img)
        rmse_value = rmse(src_img, dst_img)
        psnr_value = psnr(src_img, dst_img)
        ssim_value = ssim(src_img, dst_img)
        ms_ssim_value = msssim(src_img, dst_img)
        lpips_value = lpips_loss(sr, hr)

        total_mse_value[i] += mse_value
        total_rmse_value[i] += rmse_value
        total_psnr_value[i] += psnr_value
        total_ssim_value[i] += ssim_value[0]
        total_ms_ssim_value[i] += ms_ssim_value.real
        total_lpips_value[i] += lpips_value.item()

    os.remove("lr.bmp")
    os.remove("hr.bmp")
    os.remove("sr.bmp")

for i, title in enumerate(["raw high resolution image",
                           "nearest neighbor interpolation",
                           "bilinear interpolation",
                           "bicubic interpolation"]):
    avg_mse_value = total_mse_value[i] / len(dataloader)
    avg_rmse_value = total_rmse_value[i] / len(dataloader)
    avg_psnr_value = total_psnr_value[i] / len(dataloader)
    avg_ssim_value = total_ssim_value[i] / len(dataloader)
    avg_ms_ssim_value = total_ms_ssim_value[i] / len(dataloader)
    avg_lpips_value = total_lpips_value[i] / len(dataloader)

    print("\n=== Performance summary with " + title + " (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
          "Avg MSE: {:.2f}\n".format(avg_mse_value) +
          "Avg RMSE: {:.2f}\n".format(avg_rmse_value) +
          "Avg PSNR: {:.2f}\n".format(avg_psnr_value) +
          "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
          "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
          "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))
