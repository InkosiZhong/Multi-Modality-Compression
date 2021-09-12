from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from data_loader.datasets import Dataset
import torch
import pdb
from torch.utils.data import DataLoader


class Datasets(Dataset):
    def __init__(self, data_dir, channels=3):
        self.data_dir = data_dir
        self.channels = channels

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        if self.channels == 1:
            image = Image.open(image_ori).convert('L')
        else:
            image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(self.image_size),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)


def build_dataset(rgb_dir, ir_dir, batch_size, num_workders):
    n = 0
    if rgb_dir is not None:
        rgb_dataset = Datasets(rgb_dir)
        rgb_loader = DataLoader(dataset=rgb_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workders)
        print("rgb dataset: ", len(rgb_dataset))
        n = len(rgb_dataset) // batch_size
    else:
        rgb_loader = None
    
    if ir_dir is not None:
        ir_dataset = Datasets(ir_dir, channels=1)
        ir_loader = DataLoader(dataset=ir_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workders)
        print("ir dataset: ", len(ir_dataset))
        n = len(ir_dataset) // batch_size
    else:
        ir_loader = None

    return rgb_loader, ir_loader, n

if __name__ == '__main__':
    build_dataset()
