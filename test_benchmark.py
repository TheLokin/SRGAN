import os
import cv2
import lpips
import torch
import argparse
import torchvision.utils as utils
import torchvision.transforms as transforms

from tqdm import tqdm
from models import Generator
from dataset import DatasetFromFolder
from torch.utils.data import DataLoader
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

# Create the necessary folders
if not os.path.exists("test"):
    os.makedirs("test")

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
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (input, target) in progress_bar:
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(lr)

    utils.save_image(lr, os.path.join("test", str(i) + "_lr.bmp"))
    utils.save_image(hr, os.path.join("test", str(i) + "_hr.bmp"))
    utils.save_image(sr, os.path.join("test", str(i) + "_sr.bmp"))

    lr_img = cv2.imread(os.path.join("test", str(i) + "_lr.bmp"))
    dst_img = cv2.imread(os.path.join("test", str(i) + "_hr.bmp"))
    src_img = cv2.imread(os.path.join("test", str(i) + "_sr.bmp"))

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

    sr = cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_NEAREST)
    sr = transforms.ToTensor()(sr).unsqueeze(0)
    sr = sr.to(device)

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

    sr = cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_LINEAR)
    sr = transforms.ToTensor()(sr).unsqueeze(0)
    sr = sr.to(device)

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

    sr = cv2.resize(lr_img, (opt.crop_size, opt.crop_size),
                    interpolation=cv2.INTER_CUBIC)
    sr = transforms.ToTensor()(sr).unsqueeze(0)
    sr = sr.to(device)

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

    progress_bar.set_description(
        "[" + str(i + 1) + "/ " + str(len(dataloader)) + "]")

avg_mse_value = total_mse_value[0] / len(dataloader)
avg_rmse_value = total_rmse_value[0] / len(dataloader)
avg_psnr_value = total_psnr_value[0] / len(dataloader)
avg_ssim_value = total_ssim_value[0] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[0] / len(dataloader)
avg_lpips_value = total_lpips_value[0] / len(dataloader)

print("\n=== Performance summary with raw high resolution image (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
      "Avg MSE: {:.2f}\n".format(avg_mse_value) +
      "Avg RMSE: {:.2f}\n".format(avg_rmse_value) +
      "Avg PSNR: {:.2f}\n".format(avg_psnr_value) +
      "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
      "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))

avg_mse_value = total_mse_value[1] / len(dataloader)
avg_rmse_value = total_rmse_value[1] / len(dataloader)
avg_psnr_value = total_psnr_value[1] / len(dataloader)
avg_ssim_value = total_ssim_value[1] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[1] / len(dataloader)
avg_lpips_value = total_lpips_value[1] / len(dataloader)

print("\n=== Performance summary with nearest neighbor interpolation (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
      "Avg MSE: {:.2f}\n".format(avg_mse_value) +
      "Avg RMSE: {:.2f}\n".format(avg_rmse_value) +
      "Avg PSNR: {:.2f}\n".format(avg_psnr_value) +
      "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
      "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))

avg_mse_value = total_mse_value[2] / len(dataloader)
avg_rmse_value = total_rmse_value[2] / len(dataloader)
avg_psnr_value = total_psnr_value[2] / len(dataloader)
avg_ssim_value = total_ssim_value[2] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[2] / len(dataloader)
avg_lpips_value = total_lpips_value[2] / len(dataloader)

print("\n=== Performance summary with bilinear interpolation (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
      "Avg MSE: {:.2f}\n".format(avg_mse_value) +
      "Avg RMSE: {:.2f}\n".format(avg_rmse_value) +
      "Avg PSNR: {:.2f}\n".format(avg_psnr_value) +
      "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
      "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))

avg_mse_value = total_mse_value[3] / len(dataloader)
avg_rmse_value = total_rmse_value[3] / len(dataloader)
avg_psnr_value = total_psnr_value[3] / len(dataloader)
avg_ssim_value = total_ssim_value[3] / len(dataloader)
avg_ms_ssim_value = total_ms_ssim_value[3] / len(dataloader)
avg_lpips_value = total_lpips_value[3] / len(dataloader)

print("\n=== Performance summary with bicubic interpolation (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
      "Avg MSE: {:.2f}\n".format(avg_mse_value) +
      "Avg RMSE: {:.2f}\n".format(avg_rmse_value) +
      "Avg PSNR: {:.2f}\n".format(avg_psnr_value) +
      "Avg SSIM: {:.4f}\n".format(avg_ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(avg_ms_ssim_value) +
      "Avg LPIPS: {:.4f}\n".format(avg_lpips_value))
