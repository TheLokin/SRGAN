import os
import torch
import torch.nn as nn
import torch.optim as optim


def load_checkpoint(model, optimizer, file):
    if os.path.isfile(file):
        print("[*] Loading checkpoint '" + file + "'.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        model.load_state_dict(checkpoint["model"])
        print("[*] Loaded checkpoint '" + file +
              "' (epoch " + str(epoch) + ").")

        return epoch
    else:
        print("[!] No checkpoint found at '" + file + "'.")

        return 0
