import os
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(DatasetFromFolder, self).__init__()

        crop_size -= crop_size % upscale_factor

        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size // upscale_factor,
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        self.dataset = [Image.open(os.path.join(dataset_dir, x))
                        for x in os.listdir(dataset_dir)]

    def __getitem__(self, index):
        hr_image = self.hr_transform(self.dataset[index])
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.dataset)
