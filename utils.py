import os
import torch
import torch.nn as nn
import torch.optim as optim


def check_extension(filename):
    return any(filename.endswith(extension) for extension in ["bmp", ".png", ".jpg", ".jpeg", ".png", ".PNG", ".jpeg", ".JPEG"])


def load_checkpoint(model, optimizer, file):
    if os.path.isfile(file):
        print(f"[*] Loading checkpoint '{file}''.")
        checkpoint = torch.load(file)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[*] Loaded checkpoint '{file}'' (epoch {checkpoint['epoch']})")
    else:
        print(f"[!] no checkpoint found at '{file}'")
