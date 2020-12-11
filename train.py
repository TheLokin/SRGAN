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
parser.add_argument("--upscale-factor", type=int, default=2, choices=[2, 4], metavar="N",
                    help="Low to high resolution scaling factor (default: 2).")
parser.add_argument("--input", type=str, metavar="N",
                    help="Folder with the input dataset images.")
parser.add_argument("--target", type=str, metavar="N",
                    help="Folder with the target dataset images.")
parser.add_argument("--epoch-psnr", type=int, default=20, metavar="N",
                    help="The number of iterations is need in the training of PSNR model (default: 20).")
parser.add_argument("--checkpoint-psnr", type=int, default=0, metavar="N",
                    help="Continue with previous check point in the training of PSNR model (default: 0).")
parser.add_argument("--epoch", type=int, default=100, metavar="N",
                    help="The number of iterations is need in the training of SRGAN model (default: 100).")
parser.add_argument("--checkpoint", type=int, default=0, metavar="N",
                    help="Continue with previous check point in the training of SRGAN model (default: 0).")
parser.add_argument("--output", type=bool, default=True, metavar="N",
                    help="Generate an output image every 5000 iterations (default: false).")
opt = parser.parse_args()

# Create the necessary folders
for path in [os.path.join("weight", "SRResNet", str(opt.upscale_factor) + "x"),
             os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x"),
             os.path.join("output", "SRResNet", str(opt.upscale_factor) + "x"),
             os.path.join("output", "SRGAN", str(opt.upscale_factor) + "x",)]:
    if not os.path.exists(path):
        os.makedirs(path)

# Show warning message
if not torch.cuda.is_available():
    device = "cpu"
    print("[!] Using CPU.")
else:
    device = "cuda:0"

# Load dataset
dataset = DatasetFromFolder(opt.input, opt.target)
dataloader = DataLoader(dataset, pin_memory=True)

# Construct network architecture model of generator and discriminator
netG = Generator(16, opt.upscale_factor).to(device)
netD = Discriminator().to(device)

# We use VGG as our feature extraction method by default
content_criterion = ContentLoss().to(device)

# Perceptual loss = content loss + 1e-3 * adversarial loss
mse_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)

# Set the all model to training mode
netD.train()
netG.train()

# Define PSNR model optimizer
optimizer = torch.optim.Adam(netG.parameters())

if opt.checkpoint_psnr > 0:
    print("[*] Found PSNR pretrained model weights. Skip pre-train.")

    # Loading PSNR pre training model
    load_checkpoint(netG, optimizer, os.path.join("weight", "SRResNet", str(opt.upscale_factor) + "x",
                                                  "SRResNet_" + str(opt.upscale_factor) + "x_checkpoint-" + str(opt.checkpoint_psnr) + ".pth"))
else:
    try:
        # Pre-train generator using raw MSE loss
        print("[!] Not found pretrained weights. Start training PSNR model.")

        # Writer train PSNR model log
        with open(os.path.join("weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x_Loss.csv"), "w+") as file:
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

                progress_bar.set_description("[" + str(epoch) + "/" + str(opt.epoch_psnr) + "][" + str(
                    i + 1) + "/" + str(len(dataloader)) + "] MSE loss: {:.6f}".format(mse_loss.item()))

                # The image is saved every 5000 iterations
                if opt.output:
                    total_iter = i + (epoch - 1) * len(dataloader)
                    if (total_iter + 1) % 5000 == 0:
                        utils.save_image(lr, os.path.join(
                            "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_iter-" + str(total_iter + 1) + "_lr.bmp"))
                        utils.save_image(hr, os.path.join(
                            "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_iter-" + str(total_iter + 1) + "_hr.bmp"))
                        utils.save_image(sr, os.path.join(
                            "output", "SRResNet", str(opt.upscale_factor) + "x", "SRResNet_iter-" + str(total_iter + 1) + "_sr.bmp"))

            # The model is saved every 1 epoch
            torch.save({"epoch": epoch,
                        "optimizer": optimizer.state_dict(),
                        "state_dict": netG.state_dict()
                        }, os.path.join("weight", "SRResNet", str(opt.upscale_factor) + "x",
                                        "SRResNet_" + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))

            # Writer training log
            with open(os.path.join("weight", "SRResNet", "SRResNet_" + str(opt.upscale_factor) + "x_Loss.csv"), "a+") as file:
                writer = csv.writer(file)
                writer.writerow([epoch, avg_loss / len(dataloader)])

        print("[*] Training PSNR model done!")
    except KeyboardInterrupt:
        print(
            "[*] Training PSNR model interrupt! Saving PSNR model in epoch " + str(epoch) + ".")
        torch.save({"epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "state_dict": netG.state_dict()
                    }, os.path.join("weight", "SRResNet", str(opt.upscale_factor) + "x",
                                    "SRResNet_" + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))
        sys.exit()

# Alternating training SRGAN network
step_size = max(1, int((opt.checkpoint + opt.epoch) // len(dataloader)))

optimizerD = optim.Adam(netD.parameters())
optimizerG = optim.Adam(netG.parameters())
schedulerD = optim.lr_scheduler.StepLR(
    optimizerD, step_size=step_size, gamma=0.1)
schedulerG = optim.lr_scheduler.StepLR(
    optimizerG, step_size=step_size, gamma=0.1)

# Loading SRGAN checkpoint
if opt.checkpoint > 0:
    load_checkpoint(netG, optimizerG, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                                   "netG_" + str(opt.upscale_factor) + "x_checkpoint-" + str(opt.checkpoint) + ".pth"))
    load_checkpoint(netD, optimizerD, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                                   "netD_" + str(opt.upscale_factor) + "x_checkpoint-" + str(opt.checkpoint) + ".pth"))

# Writer train SRGAN model log
if opt.checkpoint == 0:
    with open(os.path.join("weight", "SRGAN", "SRGAN_" + str(opt.upscale_factor) + "x_Loss.csv"), "w+") as file:
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
            if opt.output:
                total_iter = i + (epoch - 1) * len(dataloader)
                if (total_iter + 1) % 5000 == 0:
                    utils.save_image(lr, os.path.join(
                        "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_iter-" + str(total_iter + 1) + "_lr.bmp"))
                    utils.save_image(hr, os.path.join(
                        "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_iter-" + str(total_iter + 1) + "_hr.bmp"))
                    utils.save_image(sr, os.path.join(
                        "output", "SRGAN", str(opt.upscale_factor) + "x", "SRGAN_iter-" + str(total_iter + 1) + "_sr.bmp"))

        # The model is saved every 1 epoch
        torch.save({"epoch": epoch,
                    "optimizer": optimizerD.state_dict(),
                    "state_dict": netD.state_dict()
                    }, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                    "netD_" + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))
        torch.save({"epoch": epoch,
                    "optimizer": optimizerG.state_dict(),
                    "state_dict": netG.state_dict()
                    }, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                    "netG_" + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))

        # Writer training log
        with open(os.path.join("weight", "SRGAN", "SRGAN_" + str(opt.upscale_factor) + "x_Loss.csv"), "a+") as file:
            writer = csv.writer(file)
            writer.writerow(
                [epoch, d_avg_loss / len(dataloader), g_avg_loss / len(dataloader)])

    print("[*] Training SRGAN model done!")
except KeyboardInterrupt:
    print("[*] Training SRGAN model interrupt! Saving SRGAN model in epoch " + str(epoch) + ".")
    torch.save({"epoch": epoch,
                "optimizer": optimizerD.state_dict(),
                "state_dict": netD.state_dict()
                }, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                "netD_ " + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))
    torch.save({"epoch": epoch,
                "optimizer": optimizerG.state_dict(),
                "state_dict": netG.state_dict()
                }, os.path.join("weight", "SRGAN", str(opt.upscale_factor) + "x",
                                "netG_ " + str(opt.upscale_factor) + "x_checkpoint-" + str(epoch) + ".pth"))
    sys.exit()
