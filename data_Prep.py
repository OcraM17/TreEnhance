import torch
import torch.utils
import torch.utils.data
import os
from torchvision import transforms
from random import random, sample
from PIL import Image


class Dataset_LOL(torch.utils.data.Dataset):
    def __init__(self, raw_dir, exp_dir, subset_img=None, size=None, training=True):
        self.raw_dir = raw_dir
        self.exp_dir = exp_dir
        self.subset_img = subset_img
        self.size = size

        if subset_img is not None:
            self.listfile = sample(os.listdir(raw_dir), self.subset_img)
        else:
            self.listfile = os.listdir(raw_dir)

        transformation = []
        if training:
            transformation.append(transforms.RandomHorizontalFlip(0.5))
            if size is not None:
                if random() > 0.5:
                    transformation.append(transforms.RandomResizedCrop((size, size)))
        if size is not None:
            transformation.append(transforms.Resize((size, size)))

        self.transforms = transforms.Compose(transformation)

    def __len__(self):
        return len(self.listfile)

    def __getitem__(self, index):
        raw = transforms.ToTensor()(Image.open(self.raw_dir + self.listfile[index]))
        expert = transforms.ToTensor()(Image.open(self.exp_dir + self.listfile[index]))
        if raw.shape != expert.shape:
            raw = transforms.Resize((self.size, self.size))(raw)
            expert = transforms.Resize((self.size, self.size))(expert)
        raw_exp = self.transforms(torch.stack([raw, expert]))
        return raw_exp[0], raw_exp[1]


class LoadDataset(torch.utils.data.Dataset):
    def __init__(self, raw_list, prob_list, win):
        self.raw_list = raw_list
        self.prob_list = prob_list
        self.win = win
        self.indices = []

    def __len__(self):
        return len(self.raw_list)

    def __getitem__(self, index):
        return self.raw_list[index], self.prob_list[index], torch.tensor(self.win[index])

