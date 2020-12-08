import os
import sys
import csv
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils

from tqdm import tqdm
from torch.utils.data import DataLoader

from loss import ContentLoss
from utils import load_checkpoint
from dataset import DatasetFromFolder
from models import Generator, Discriminator


parser = argparse.ArgumentParser(
    description="Photo-Realistic Single Image Super-Resolution.")
parser.add_argument("--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default: 4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4],
                    help="Low to high resolution scaling factor. (default: 4).")
parser.add_argument("--epoch-psnr", default=1, type=int, metavar="N",
                    help="The number of iterations is need in the training of PSNR model. (default: 1e6)")
parser.add_argument("--epoch", default=1, type=int, metavar="N",
                    help="The number of iterations is need in the training of SRGAN model. (default: 2e5)")
parser.add_argument("--checkpoint", default=0, type=int, metavar="N",
                    help="Continue with previous check point. (default: -1)")
parser.add_argument("--output", default=False, type=bool, metavar="N",
                    help="Generate an output image every 5000 iterations. (default: false)")
opt = parser.parse_args()

# Create the necessary folders
for path in [os.path.join("weight", "SRResNet"),
             os.path.join("weight", "SRGAN"),
             os.path.join(
                 "output", f"{opt.upscale_factor}x", "SRResNet", "lr"),
             os.path.join(
                 "output", f"{opt.upscale_factor}x", "SRResNet", "hr"),
             os.path.join(
                 "output", f"{opt.upscale_factor}x", "SRResNet", "sr"),
             os.path.join("output", f"{opt.upscale_factor}x", "SRGAN", "lr"),
             os.path.join("output", f"{opt.upscale_factor}x", "SRGAN", "hr"),
             os.path.join("output", f"{opt.upscale_factor}x", "SRGAN", "sr")]:
    if not os.path.exists(path):
        os.makedirs(path)

# Show warning message
if not torch.cuda.is_available():
    device = "cpu"
    print("[!] Using CPU.")
else:
    device = "cuda:0"

# Load dataset
dataset = DatasetFromFolder(os.path.join("data", str(opt.upscale_factor) + "x", "train",
                                         "input"), os.path.join("data", str(opt.upscale_factor) + "x", "train", "target"))
dataloader = DataLoader(dataset, pin_memory=True, num_workers=int(opt.workers))

# Construct network architecture model of generator and discriminator
netG = Generator(16, opt.upscale_factor).to(device)
netD = Discriminator().to(device)

# We use VGG5.4 as our feature extraction method by default
content_criterion = ContentLoss().to(device)

# Perceptual loss = content loss + 1e-3 * adversarial loss
mse_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)

# Set the all model to training mode
netD.train()
netG.train()

# Define PSNR model optimizer
optimizer = torch.optim.Adam(netG.parameters())

if opt.checkpoint > 0:
    print("[*] Found PSNR pretrained model weights. Skip pre-train.")

    # Loading PSNR pre training model
    load_checkpoint(netG, optimizer, os.path.join(
        "weight", "SRResNet", f"SRResNet_{opt.upscale_factor}x_checkpoint-{opt.epoch_psnr}.pth"))
else:
    try:
        # Pre-train generator using raw MSE loss
        print("[!] Not found pretrained weights. Start training PSNR model.")

        # Writer train PSNR model log
        with open(os.path.join("output", f"{opt.upscale_factor}x", "SRResNet_Loss.csv"), "w+") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "MSE Loss"])

        for epoch in range(1, opt.epoch_psnr + 1):
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

                progress_bar.set_description(
                    f"[{epoch}/{opt.epoch_psnr}][{i + 1}/{len(dataloader)}] MSE loss: {mse_loss.item():.6f}")

                # The image is saved every 5000 iterations
                if opt.output:
                    total_iter = i + (epoch - 1) * len(dataloader)
                    if (total_iter + 1) % 5000 == 0:
                        utils.save_image(lr, os.path.join(
                            "output", f"{opt.upscale_factor}x", "SRResNet", "lr", f"SRResNet_{total_iter + 1}.bmp"))
                        utils.save_image(hr, os.path.join(
                            "output", f"{opt.upscale_factor}x", "SRResNet", "hr", f"SRResNet_{total_iter + 1}.bmp"))
                        utils.save_image(sr, os.path.join(
                            "output", f"{opt.upscale_factor}x", "SRResNet", "sr", f"SRResNet_{total_iter + 1}.bmp"))

            # The model is saved every 1 epoch
            torch.save({"epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "state_dict": netG.state_dict()
                        }, os.path.join("weight", "SRResNet", f"SRResNet_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))

            # Writer training log
            with open(os.path.join("output", f"{opt.upscale_factor}x", "SRResNet_Loss.csv"), "a+") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, avg_loss / len(dataloader)])

        print("[*] Training PSNR model done!")
    except KeyboardInterrupt:
        print(
            f"[*] Training PSNR model interrupt! Saving PSNR model in epoch {epoch}.")
        torch.save({"epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": netG.state_dict()
                    }, os.path.join("weight", "SRResNet", f"SRResNet_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))
        sys.exit()

# Alternating training SRGAN network
optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())
schedulerD = optim.lr_scheduler.StepLR(
    optimizerD, step_size=opt.epoch // 2, gamma=0.1)
schedulerG = optim.lr_scheduler.StepLR(
    optimizerG, step_size=opt.epoch // 2, gamma=0.1)

# Loading SRGAN checkpoint
if opt.checkpoint > 0:
    load_checkpoint(netG, optimizerG, os.path.join(
        "weight", "SRGAN", f"netG_{opt.upscale_factor}x_checkpoint-{opt.checkpoint}.pth"))
    load_checkpoint(netD, optimizerD, os.path.join(
        "weight", "SRGAN", f"netD_{opt.upscale_factor}x_checkpoint-{opt.checkpoint}.pth"))

# Writer train SRGAN model log
if opt.checkpoint == 0:
    with open(os.path.join("output", f"{opt.upscale_factor}x", "SRGAN_Loss.csv"), "w+") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "D Loss", "G Loss"])

try:
    # Train SRGAN model
    print("[*] Starting training SRGAN model!")

    for epoch in range(1 + max(0, opt.checkpoint), opt.epoch + 1 + max(opt.checkpoint, 0)):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        g_avg_loss = 0
        d_avg_loss = 0

        for i, (input, target) in progress_bar:
            # Generate data
            lr = input.to(device)
            hr = target.to(device)

            batch_size = lr.size(0)
            real_label = torch.ones(batch_size, dtype=lr.dtype, device=device)
            fake_label = torch.zeros(batch_size, dtype=lr.dtype, device=device)

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

            progress_bar.set_description(f"[{epoch}/{opt.epoch}][{i + 1}/{len(dataloader)}] "
                                         f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                         f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z1:.6f}/{D_G_z2:.6f}")

            # The image is saved every 5000 iterations
            if opt.output:
                total_iter = i + (epoch - 1) * len(dataloader)
                if (total_iter + 1) % 5000 == 0:
                    utils.save_image(lr, os.path.join(
                        "output", f"{opt.upscale_factor}x", "SRGAN", "lr", f"SRGAN_{total_iter + 1}.bmp"))
                    utils.save_image(hr, os.path.join(
                        "output", f"{opt.upscale_factor}x", "SRGAN", "hr", f"SRGAN_{total_iter + 1}.bmp"))
                    utils.save_image(sr, os.path.join(
                        "output", f"{opt.upscale_factor}x", "SRGAN", "sr", f"SRGAN_{total_iter + 1}.bmp"))

        # The model is saved every 1 epoch
        torch.save({"epoch": epoch,
                    "optimizer": optimizerD.state_dict(),
                    "state_dict": netD.state_dict()
                    }, os.path.join("weight", "SRGAN", f"netD_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))
        torch.save({"epoch": epoch,
                    "optimizer": optimizerG.state_dict(),
                    "state_dict": netG.state_dict()
                    }, os.path.join("weight", "SRGAN", f"netG_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))

        # Writer training log
        with open(os.path.join("output", f"{opt.upscale_factor}x", "SRGAN_Loss.csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

    print("[*] Training SRGAN model done!")
except KeyboardInterrupt:
    print(
        f"[*] Training SRGAN model interrupt! Saving SRGAN model in epoch {epoch}.")
    torch.save({"epoch": epoch,
                "optimizer": optimizerD.state_dict(),
                "state_dict": netD.state_dict()
                }, os.path.join("weight", "SRGAN", f"netD_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))
    torch.save({"epoch": epoch,
                "optimizer": optimizerG.state_dict(),
                "state_dict": netG.state_dict()
                }, os.path.join("weight", "SRGAN", f"netG_{opt.upscale_factor}x_checkpoint-{epoch}.pth"))
    sys.exit()
