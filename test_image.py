import os
import cv2
import lpips
import torch
import argparse
import torchvision.utils as utils
import torchvision.transforms as transforms

from PIL import Image
from models import Generator
from sewar.full_ref import mse, rmse, psnr, ssim, msssim


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution Test.")
parser.add_argument("--lr-image", type=str, metavar="N",
                    help="Low resolution image.")
parser.add_argument("--hr-image", type=str, metavar="N",
                    help="High resolution image.")
parser.add_argument("--nn-image", type=str, metavar="N",
                    help="Nearest neighbor interpolation image.")
parser.add_argument("--bl-image", type=str, metavar="N",
                    help="Bilinear interpolation image.")
parser.add_argument("--bc-image", type=str, metavar="N",
                    help="Bicubic interpolation image.")
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

# Construct SRGAN model
model = Generator(16, opt.upscale_factor).to(device)
checkpoint = torch.load(os.path.join(
    "weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"), map_location=device)
model.load_state_dict(checkpoint["model"])

# Set model eval mode
model.eval()

# Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
lpips_loss = lpips.LPIPS(net="vgg").to(device)

lr = Image.open(opt.lr_image)
hr = Image.open(opt.hr_image)
lr = transforms.ToTensor()(lr).unsqueeze(0)
hr = transforms.ToTensor()(hr).unsqueeze(0)
lr = lr.to(device)
hr = hr.to(device)

with torch.no_grad():
    sr = model(lr)

utils.save_image(sr, os.path.join("test", "SRGAN_sr.bmp"))

dst_img = cv2.imread(opt.hr_image)
src_img = cv2.imread(os.path.join("test", "SRGAN_sr.bmp"))

mse_value = mse(src_img, dst_img)
rmse_value = rmse(src_img, dst_img)
psnr_value = psnr(src_img, dst_img)
ssim_value = ssim(src_img, dst_img)[0]
ms_ssim_value = msssim(src_img, dst_img).real
lpips_value = lpips_loss(sr, hr).item()

print("\n=== Performance summary\n" +
      "Avg MSE: {:.4f}\n".format(mse_value) +
      "Avg RMSE: {:.4f}\n".format(rmse_value) +
      "Avg PSNR: {:.4f}\n".format(psnr_value) +
      "Avg SSIM: {:.4f}\n".format(ssim_value) +
      "Avg MS-SSIM: {:.4f}\n".format(ms_ssim_value) +
      "Avg LPIPS: {:.4f}".format(lpips_value))

if os.path.exists(opt.nn_image):
    src_img = cv2.imread(opt.nn_image)
    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)[0]
    ms_ssim_value = msssim(src_img, dst_img).real
    lpips_value = lpips_loss(sr, hr).item()

    print("\n=== Performance summary nearest neighbor (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
          "Avg MSE: {:.4f}\n".format(mse_value) +
          "Avg RMSE: {:.4f}\n".format(rmse_value) +
          "Avg PSNR: {:.4f}\n".format(psnr_value) +
          "Avg SSIM: {:.4f}\n".format(ssim_value) +
          "Avg MS-SSIM: {:.4f}\n".format(ms_ssim_value) +
          "Avg LPIPS: {:.4f}".format(lpips_value))

if os.path.exists(opt.bl_image):
    src_img = cv2.imread(opt.bl_image)
    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)[0]
    ms_ssim_value = msssim(src_img, dst_img).real
    lpips_value = lpips_loss(sr, hr).item()

    print("\n=== Performance summary bilinear (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
          "Avg MSE: {:.4f}\n".format(mse_value) +
          "Avg RMSE: {:.4f}\n".format(rmse_value) +
          "Avg PSNR: {:.4f}\n".format(psnr_value) +
          "Avg SSIM: {:.4f}\n".format(ssim_value) +
          "Avg MS-SSIM: {:.4f}\n".format(ms_ssim_value) +
          "Avg LPIPS: {:.4f}".format(lpips_value))

if os.path.exists(opt.bc_image):
    src_img = cv2.imread(opt.bc_image)
    sr = transforms.ToTensor()(src_img).unsqueeze(0)
    sr = sr.to(device)

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)[0]
    ms_ssim_value = msssim(src_img, dst_img).real
    lpips_value = lpips_loss(sr, hr).item()

    print("\n=== Performance summary bicubic (upsampling x" + str(opt.upscale_factor) + ")" + "\n" +
          "Avg MSE: {:.4f}\n".format(mse_value) +
          "Avg RMSE: {:.4f}\n".format(rmse_value) +
          "Avg PSNR: {:.4f}\n".format(psnr_value) +
          "Avg SSIM: {:.4f}\n".format(ssim_value) +
          "Avg MS-SSIM: {:.4f}\n".format(ms_ssim_value) +
          "Avg LPIPS: {:.4f}".format(lpips_value))
