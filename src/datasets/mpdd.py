import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

MPDD_CLASSES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]
class MPDD(Dataset):
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
        """Dataset for MPDD.
        Args:
            data_root: Root directory of the MPDD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'bracket_black'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            transform: Optional torchvision transforms to apply to the images.
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
            cls_label: If True, include class label in the output.
            anom_only: If True, only include anomalous samples.
            normal_only: If True, only include normal samples.
            **kwargs: Additional keyword arguments.
        """
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
        
        # # load files from the dataset
        self.img_files = self.get_files()
        self.labels = [0] * len(self.img_files)
        if self.split == 'test':
            def mask_to_tensor(img):
                return torch.from_numpy(np.array(img, dtype=np.uint8)).long()
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res, interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(mask_to_tensor)
                ]
            )

            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)

            self.normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.anom_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_classes = len(MPDD_CLASSES)
        
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
        
        cls_name = str(img_file).split("/")[-4]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsnames"] = cls_name
        inputs["clslabels"] = MPDD_CLASSES.index(cls_name)
        inputs["filenames"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["samples"] = sample
            return inputs
        else:
            inputs["samples"] = sample
            inputs["labels"] = label
            if "good" in str(img_file):
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = str(img_file).split("/")[-2]
            if self.is_mask:
                mask_dir =  img_file.parent.parent.parent / 'ground_truth' / img_file.parent.name 
                mask_file = mask_dir / img_file.name.replace('.png', '_mask.png')
                if 'good' == img_file.parent.name:
                    mask = Image.new('L', (self.input_res, self.input_res), 0)
                else:
                    with open(mask_file, 'rb') as f:
                        mask = Image.open(f)
                        mask = mask.convert('L')
                mask = self.mask_transform(mask)
                inputs["masks"] = mask
            return inputs
                
    def __len__(self):
        if self.anom_only:
            return len(self.anom_indices)
        elif self.normal_only:
            return len(self.normal_indices)
        else:
            return len(self.img_files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            anomalous_img_files = []
            anomalous_dirs = sorted(Path(os.path.join(self.data_root, self.category, 'test')).glob('*'))
            for anomalous_dir in anomalous_dirs:
                if "good" in str(anomalous_dir):
                    continue
                anomalous_img_files += sorted(anomalous_dir.glob('*.png'))
            anomalous_img_files = sorted(anomalous_img_files)
            
            files = normal_img_files + anomalous_img_files
        return files

if __name__ == "__main__":
    data_root = "/home/haselab/projects/sakai_ssd/InversionAD/data/mpdd"
    category = "bracket_black"
    input_res = 224
    split = "test"
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res)),
        transforms.ToTensor(),
    ])
    dataset = MPDD(data_root=data_root, category=category, input_res=input_res, split=split, transform=transform, is_mask=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch["samples"].shape)  # Should print the shape of the images
        print(batch["labels"])          # Should print the labels
        if "masks" in batch:
            print(batch["masks"].shape)  # Should print the shape of the masks if available
        break
    # Example usage of the MPDD dataset class