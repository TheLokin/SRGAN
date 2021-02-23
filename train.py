import os
import sys
import csv
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

from tqdm import tqdm
from loss import ContentLoss
from pytorch_msssim import SSIM
from utils import load_checkpoint
from torch.utils.data import DataLoader
from dataset import TrainDatasetFromFolder
from models import Generator, Discriminator


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution Training.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--crop-size", type=int, default=200, metavar="N",
                    help="Crop size for the training images (default: 200).")
parser.add_argument("--upscale-factor", type=int, default=2, metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--epoch-SRResNet", type=int, default=1000, metavar="N",
                    help="The number of iterations is need in the training of SRResNet model (default: 1000).")
parser.add_argument("--epoch", type=int, default=5000, metavar="N",
                    help="The number of iterations is need in the training of SRGAN model (default: 5000).")
parser.add_argument("--ssim-loss", type=str2bool, nargs="?", const=True, default=False,
                    help="Use SSIM as loss function (default: False).")
opt = parser.parse_args()

target_size = opt.crop_size * opt.upscale_factor

# Create the necessary folders
for path in [os.path.join("weight", "SRResNet"),
             os.path.join("weight", "SRGAN"),
             os.path.join("output", "SRResNet", str(opt.upscale_factor) + "x"),
             os.path.join("output", "SRGAN", str(opt.upscale_factor) + "x")]:
    if not os.path.exists(path):
        os.makedirs(path)

# Show warning message
if not torch.cuda.is_available():
    device = "cpu"
    print("[!] Using CPU.")
else:
    device = "cuda:0"

# Load dataset
dataset = TrainDatasetFromFolder(
    opt.dataset, target_size, opt.upscale_factor)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct network architecture model of generator and discriminator
netG = Generator(16, opt.upscale_factor).to(device)
netD = Discriminator().to(device)

# Set the all model to training mode
netD.train()
netG.train()

# We use VGG as our feature extraction method by default
if opt.ssim_loss:
    content_criterion = SSIM(data_range=255, size_average=True, channel=3)
else:
    content_criterion = ContentLoss().to(device)

# Perceptual loss = content loss + 1e-3 * adversarial loss
mse_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)

# Define PSNR model optimizer
optimizer = optim.Adam(netG.parameters())

# Loading PSNR pre training model
print("[*] Start training SRResNet model.")
checkpoint = load_checkpoint(netG, optimizer, os.path.join(
    "weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

# Writer train PSNR model log
if checkpoint == 0:
    with open(os.path.join("weight", "SRResNet", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "MSE Loss"])

# Pre-train generator using raw MSE loss
for epoch in range(checkpoint + 1, opt.epoch_SRResNet + 1):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    avg_loss = 0

    for i, (input, target) in progress_bar:
        # Set generator gradients to zero
        netG.zero_grad()

        # Generate data
        lr = input.to(device)
        hr = target.to(device)

        # Generating fake high resolution images from real low resolution images
        sr = netG(lr)

        # The MSE of the generated fake high-resolution image and real high-resolution image is calculated
        mse_loss = mse_criterion(sr, hr)

        # Calculate gradients for generator
        mse_loss.backward()

        # Update generator weights
        optimizer.step()

        avg_loss += mse_loss.item()

        progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch_SRResNet) + "][" + str(
            i + 1) + "/" + str(len(dataloader)) + "] MSE loss: {:.6f}".format(mse_loss.item()))

        # The image is saved every 5000 iterations
        total_iter = i + (epoch - 1) * len(dataloader)
        if (total_iter + 1) % 5000 == 0:
            utils.save_image(lr, os.path.join(
                "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_" + str(total_iter + 1) + "_lr.bmp"))
            utils.save_image(hr, os.path.join(
                "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_" + str(total_iter + 1) + "_hr.bmp"))
            utils.save_image(sr, os.path.join(
                "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_" + str(total_iter + 1) + "_sr.bmp"))

    # The model is saved every 1 epoch
    torch.save({"epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "model": netG.state_dict()
                }, os.path.join("weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x.pth"))

    # Writer training log
    with open(os.path.join("weight", "SRResNet", "SRResNet_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, avg_loss / len(dataloader)])

print("[*] Training PSNR model done!")

# Define SRGAN model optimizers
optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())
step_size = max(1, int(((checkpoint + opt.epoch) // len(dataloader)) // 2))
schedulerD = optim.lr_scheduler.StepLR(
    optimizerD, step_size=step_size, gamma=0.1)
schedulerG = optim.lr_scheduler.StepLR(
    optimizerG, step_size=step_size, gamma=0.1)

# Loading SRGAN checkpoint
print("[*] Starting training SRGAN model")
checkpoint = load_checkpoint(netG, optimizerG, os.path.join(
    "weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"))
checkpoint = load_checkpoint(netD, optimizerD, os.path.join(
    "weight", "SRGAN", "netD_" + str(opt.upscale_factor) + "x.pth"))

# Writer train SRGAN model log
if checkpoint == 0:
    with open(os.path.join("weight", "SRGAN", "SRGAN_Loss_" + str(opt.upscale_factor) + "x.csv"), "w+") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "D Loss", "G Loss"])

# Train generator and discriminator
for epoch in range(checkpoint + 1, opt.epoch + 1):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    g_avg_loss = 0
    d_avg_loss = 0

    for i, (input, target) in progress_bar:
        # Generate data
        lr = input.to(device)
        hr = target.to(device)

        batch_size = lr.size(0)
        real_label = torch.ones(
            batch_size, dtype=lr.dtype, device=device)
        fake_label = torch.zeros(
            batch_size, dtype=lr.dtype, device=device)

        ###############################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ###############################################################

        # Set discriminator gradients to zero
        netD.zero_grad()

        # Generate a super-resolution image
        sr = netG(lr)

        # Train with real high resolution image
        hr_output = netD(hr)
        errD_hr = adversarial_criterion(hr_output, real_label)
        errD_hr.backward()
        D_x = hr_output.mean().item()

        # Train with fake high resolution image
        sr_output = netD(sr.detach())
        errD_sr = adversarial_criterion(sr_output, fake_label)
        errD_sr.backward()
        D_G_z1 = sr_output.mean().item()
        errD = errD_hr + errD_sr
        optimizerD.step()

        ###############################################
        # (2) Update G network: maximize log(D(G(z))) #
        ###############################################

        # Set generator gradients to zero
        netG.zero_grad()

        # We then define the VGG loss as the euclidean distance between the feature representations of
        # a reconstructed image G(LR) and the reference image LR
        if opt.ssim_loss:
            content_loss = 1 - content_criterion(sr, hr)
        else:
            content_loss = content_criterion(sr, hr)

        # Second train with fake high resolution image
        sr_output = netD(sr)

        # The generative loss is defined based on the probabilities of the discriminator
        # D(G(LR)) over all training samples as
        adversarial_loss = adversarial_criterion(sr_output, real_label)

        # We formulate the perceptual loss as the weighted sum of a content loss and an adversarial loss component as
        errG = content_loss + 1e-3 * adversarial_loss
        errG.backward()
        D_G_z2 = sr_output.mean().item()
        optimizerG.step()

        # Dynamic adjustment of learning rate
        schedulerD.step()
        schedulerG.step()

        d_avg_loss += errD.item()
        g_avg_loss += errG.item()

        progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch) + "][" + str(i + 1) + "/" + str(len(dataloader)) +
                                     "] Loss_D: {:.6f} Loss_G: {:.6f} ".format(errD.item(), errG.item()) + "D(HR): {:.6f} D(G(LR)): {:.6f}/{:.6f}".format(D_x, D_G_z1, D_G_z2))

        # The image is saved every 5000 iterations
        total_iter = i + (epoch - 1) * len(dataloader)
        if (total_iter + 1) % 5000 == 0:
            utils.save_image(lr, os.path.join(
                "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_" + str(total_iter + 1) + "_lr.bmp"))
            utils.save_image(hr, os.path.join(
                "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_" + str(total_iter + 1) + "_hr.bmp"))
            utils.save_image(sr, os.path.join(
                "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_" + str(total_iter + 1) + "_sr.bmp"))

    # The model is saved every 1 epoch
    torch.save({"epoch": epoch,
                "optimizer": optimizerD.state_dict(),
                "model": netD.state_dict()
                }, os.path.join("weight", "SRGAN", "netD_" + str(opt.upscale_factor) + "x.pth"))
    torch.save({"epoch": epoch,
                "optimizer": optimizerG.state_dict(),
                "model": netG.state_dict()
                }, os.path.join("weight", "SRGAN", "netG_" + str(opt.upscale_factor) + "x.pth"))

    # Writer training log
    with open(os.path.join("weight", "SRGAN", "SRGAN_Loss_" + str(opt.upscale_factor) + "x.csv"), "a+") as file:
        writer = csv.writer(file)
        writer.writerow(
            [epoch, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

print("[*] Training SRGAN model done!")
