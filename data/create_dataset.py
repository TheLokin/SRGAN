import os
import cv2
import argparse

from tqdm import tqdm
from utils import remove_edges, crop_image, split_image, split_dataset


parser = argparse.ArgumentParser(
    description="Create a new dataset from another.")
parser.add_argument("--dataset", type=str, metavar="N",
                    help="Folder with the dataset images.")
parser.add_argument("--output", type=str, default=".", metavar="N",
                    help="Folder for the new dataset (default: .).")
opt = parser.parse_args()

# Create the necessary folders
for path in [os.path.join(opt.output, "2x", "train", "input"),
             os.path.join(opt.output, "2x", "train", "target"),
             os.path.join(opt.output, "2x", "test", "input"),
             os.path.join(opt.output, "2x", "test", "target"),
             os.path.join(opt.output, "4x", "train", "input"),
             os.path.join(opt.output, "4x", "train", "target"),
             os.path.join(opt.output, "4x", "test", "input"),
             os.path.join(opt.output, "4x", "test", "target")]:
    if not os.path.exists(path):
        os.makedirs(path)

# Preprocess dataset
number = 1
for filename in tqdm(os.listdir(opt.dataset), desc="Generating images from dataset"):
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

        cv2.imwrite(os.path.join(opt.output, "2x", "train", "input",
                                 "preprocess" + str(number) + ".bmp"), img)
        cv2.imwrite(os.path.join(opt.output, "2x", "train", "target",
                                 "preprocess" + str(number) + ".bmp"), img2x)
        cv2.imwrite(os.path.join(opt.output, "4x", "train", "input",
                                 "preprocess" + str(number) + ".bmp"), img)
        cv2.imwrite(os.path.join(opt.output, "4x", "train", "target",
                                 "preprocess" + str(number) + ".bmp"), img4x)
        number += 1

split_dataset(os.path.join(opt.output, "2x"))
split_dataset(os.path.join(opt.output, "4x"))
