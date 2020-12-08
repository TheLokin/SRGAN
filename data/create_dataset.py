import os
import cv2
import argparse

from tqdm import tqdm
from utils import remove_edges, crop_image, split_image, split_dataset


parser = argparse.ArgumentParser(description="Create dataset.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
opt = parser.parse_args()

# Create the necessary folders
for path in [os.path.join("2x", "train", "input"), os.path.join("2x", "train", "target"),
             os.path.join("2x", "test", "input"), os.path.join(
                 "2x", "test", "target"),
             os.path.join("4x", "train", "input"), os.path.join(
                 "4x", "train", "target"),
             os.path.join("4x", "test", "input"), os.path.join("4x", "test", "target")]:
    if not os.path.exists(path):
        os.makedirs(path)

# Preprocess dataset
number = 1
for filename in tqdm(os.listdir(opt.dataset), desc=f"Generating images from dataset"):
    img = cv2.imread(os.path.join(opt.dataset, filename))
    img = remove_edges(img)
    img = crop_image(img)

    for split in split_image(img):
        img = cv2.resize(split, (200, 200),
                         interpolation=cv2.INTER_CUBIC)
        img2x = cv2.resize(split, (400, 400),
                           interpolation=cv2.INTER_CUBIC)
        img4x = cv2.resize(split, (800, 800),
                           interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join("2x", "train", "input",
                                 f"preprocess{number}.bmp"), img)
        cv2.imwrite(os.path.join("2x", "train", "target",
                                 f"preprocess{number}.bmp"), img2x)
        cv2.imwrite(os.path.join("4x", "train", "input",
                                 f"preprocess{number}.bmp"), img)
        cv2.imwrite(os.path.join("4x", "train", "target",
                                 f"preprocess{number}.bmp"), img4x)
        number += 1

split_dataset("2x")
split_dataset("4x")
