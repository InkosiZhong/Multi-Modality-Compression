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
from torch.utils.data.distributed import DistributedSampler


class Datasets(Dataset):
    def __init__(self, data_dir, data_list=None, channels=3):
        self.channels = channels

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {data_dir} not exitd")

        if data_list is None:
            self.image_path = sorted(glob(os.path.join(data_dir, "*.*")))
        else:
            if not os.path.exists(data_list):
                raise Exception(f"[!] {data_list} not exitd")
            with open(data_list) as f:
                self.image_path = [data_dir + '/' + l.strip('\n') for l in f.readlines()]

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


def build_dataset(rgb_dir, ir_dir, batch_size, num_workders, dist = False, data_list = None):
    n = 0
    if rgb_dir is not None:
        rgb_dataset = Datasets(rgb_dir, data_list=data_list, channels=3)
        rgb_sampler = DistributedSampler(rgb_dataset, shuffle=False) if dist else None
        rgb_loader = DataLoader(dataset=rgb_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workders,
                                sampler=rgb_sampler)
        print("rgb dataset: ", len(rgb_dataset))
        n = len(rgb_dataset) // batch_size
    else:
        rgb_loader = None
    
    if ir_dir is not None:
        ir_dataset = Datasets(ir_dir, data_list=data_list, channels=1)
        ir_sampler = DistributedSampler(ir_dataset, shuffle=False) if dist else None
        ir_loader = DataLoader(dataset=ir_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workders,
                                sampler=ir_sampler)
        print("ir dataset: ", len(ir_dataset))
        n = len(ir_dataset) // batch_size
    else:
        ir_loader = None

    return rgb_loader, ir_loader, n