import os

from PIL import Image
from utils import check_extension
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ToTensor


class DatasetFromFolder(Dataset):
    def __init__(self, input_dir, target_dir):
        super(DatasetFromFolder, self).__init__()

        self.transforms = Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()
        ])

        self.input_filenames = [os.path.join(
            input_dir, x) for x in os.listdir(input_dir) if check_extension(x)]
        self.target_filenames = [os.path.join(
            target_dir, x) for x in os.listdir(target_dir) if check_extension(x)]

    def __getitem__(self, index):
        lr_image = self.transforms(Image.open(self.input_filenames[index]))
        hr_image = self.transforms(Image.open(self.target_filenames[index]))

        return lr_image, hr_image

    def __len__(self):
        return len(self.input_filenames)
