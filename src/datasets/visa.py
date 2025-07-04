# LICENSE-DATASET  candle  capsules  cashew  chewinggum  fryum  macaroni1  macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum  split_csv

## Meta csv example
# object,split,label,image,mask
# candle,train,normal,candle/Data/Images/Normal/0836.JPG,
# candle,train,normal,candle/Data/Images/Normal/0451.JPG,
# candle,train,normal,candle/Data/Images/Normal/0516.JPG,
# candle,train,normal,candle/Data/Images/Normal/0977.JPG,
# candle,train,normal,candle/Data/Images/Normal/0249.JPG,
# candle,train,normal,candle/Data/Images/Normal/0548.JPG,

import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

VISA_CLASSES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

class VisA(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False, 
        anom_only=False,
        normal_only=False,
        **kwargs
    ):
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.custom_transforms = transform
        self.is_mask = is_mask
        self.cls_label = cls_label
        self.anom_only = anom_only
        self.normal_only = normal_only

        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'
        
        # Load csv meta file
        csv_path = os.path.join(self.data_root, "split_csv", "1cls.csv")
        self.meta = pd.read_csv(csv_path)
        
        # Get files
        self.img_files = self.get_files()
        self.labels = [0] * len(self.img_files)
        if self.split == 'test':
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_res, self.input_res), interpolation=InterpolationMode.NEAREST),
                    transforms.ToTensor(),
                ]
            )
            
            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'Normal':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
            
            self.normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.anom_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_classes = len(VISA_CLASSES)
        
    def __len__(self):
        if self.anom_only:
            return len(self.anom_indices)
        elif self.normal_only:
            return len(self.normal_indices)
        else:
            return len(self.img_files)

    def get_files(self):
        if self.split == 'train':
            files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'train')]
        else:
            normal_img_files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'test') & (self.meta['label'] == 'normal')]
            anom_img_files = self.meta[(self.meta['object'] == self.category) & (self.meta['split'] == 'test') & (self.meta['label'] == 'anomaly')]
            files = pd.concat([normal_img_files, anom_img_files])
        files = sorted([os.path.join(self.data_root, f) for f in files['image']])
        return files

    def __getitem__(self, index):
        inputs = {}
        
        if self.anom_only:
            img_file = self.img_files[self.anom_indices[index]]
            label = self.labels[self.anom_indices[index]]
        elif self.normal_only:
            img_file = self.img_files[self.normal_indices[index]]
            label = self.labels[self.normal_indices[index]]
        else:
            img_file = self.img_files[index]
            label = self.labels[index]
        
        cls_name = img_file.split(os.path.sep)[-5]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsnames"] = cls_name
        inputs["clslabels"] = VISA_CLASSES.index(cls_name)
        inputs["filenames"] = img_file

        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["samples"] = sample
            return inputs
        else:
            inputs["samples"] = sample
            inputs["labels"] = label
            if "Normal" in str(img_file):
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = "anomaly"
            if self.is_mask:
                mask_dir =  Path(img_file).parent.parent.parent / 'Masks' / 'Anomaly'
                mask_file = mask_dir / (Path(img_file).stem + '.png')
                if inputs["anom_type"] == "good":
                    mask = Image.new('L', (self.input_res, self.input_res), 0)
                else:
                    with open(mask_file, 'rb') as f:
                        mask = Image.open(f)
                        mask = mask.convert('L')
                mask = self.mask_transform(mask).bool()
                inputs["masks"] = mask
            return inputs

if __name__ == "__main__":
    dataset_root = "/home/haselab/projects/sakai/AnoMAR/AnoMAR/data/VisA"
    category = "candle"
    input_res = 224
    split = "test"
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = VisA(dataset_root, category, input_res, split, transform)
    print(len(dataset))
    print(dataset[0])
