import os
from pathlib import Path
from typing import *

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

# audiojack       eraser     phone_battery   switch         toy_brick    usb_adaptor   zipper
# bottle_cap      fire_hood  plastic_nut     regulator          terminalblock  transistor1  vcpill
# button_battery  mint       plastic_plug    rolled_strip_base  toothbrush     u_block      wooden_beads
# end_cap         pcb        porcelain_doll  sim_card_set       toy            usb          woodstick

REALIAD_CLASSES = [
    'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
    'fire_hood', 'mint', 'phone_battery', 'plastic_nut', 'plastic_plug',
    'pcb', 'porcelain_doll', 'regulator', 'rolled_strip_base',
    'sim_card_set', 'switch', 'terminalblock', 'toothbrush',
    'toy', 'toy_brick', 'transistor1', 'u_block', 'usb', 'usb_adaptor',
    'vcpill', 'wooden_beads', 'woodstick', 'zipper', 'mounts', 'tape'
]  

NORMAL_PREFIX = 'OK'

class RealIAD(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        meta_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_mask=False, 
        cls_label=False, 
        anom_only=False,
        normal_only=False,
        **kwargs
    ):
        """Dataset for MVTec AD.
        Args:
            data_root: Root directory of MVTecAD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'hazelnut'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.meta_dir = meta_dir
        self.custom_transforms = transform
        self.is_mask = is_mask
        self.cls_label = cls_label
        self.anom_only = anom_only
        self.normal_only = normal_only
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'test'
        assert sum([self.anom_only, self.normal_only]) <= 1, "Only one of anom_only or normal_only can be True. Currently: anom_only={}, normal_only={}".format(self.anom_only, self.normal_only)
        
        # # load files from the dataset
        self.img_files, self.labels_str, self.masks = self.get_files()
        self.labels = []
        for label in self.labels_str:
            if label == NORMAL_PREFIX:
                self.labels.append(0)
            else:
                self.labels.append(1)
        if self.split == 'test':
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res, interpolation=InterpolationMode.NEAREST),
                    transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.uint8)).long())
                ]
            )
            self.normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
            self.anom_indices = [i for i, label in enumerate(self.labels) if label == 1]
        self.num_classes = len(REALIAD_CLASSES)
        
    def __getitem__(self, index):
        inputs = {}
        if self.anom_only:
            index = self.anom_indices[index]
            img_file = Path(self.img_files[index])
            label = self.labels[index]
        elif self.normal_only:
            index = self.normal_indices[index]
            img_file = Path(self.img_files[index])
            label = self.labels[index]
        else:
            img_file = Path(self.img_files[index])
            label = self.labels[index]
        
        cls_name = str(img_file).split("/")[-5]
        with open(img_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        inputs["clsnames"] = cls_name
        inputs["clslabels"] = REALIAD_CLASSES.index(cls_name)
        inputs["filenames"] = str(img_file)
        
        sample = self.custom_transforms(img)
        
        if self.split == 'train' or self.split == 'val':
            inputs["samples"] = sample
            return inputs
        else:
            inputs["samples"] = sample
            inputs["labels"] = label
            if self.labels_str[index] == NORMAL_PREFIX:
                inputs["anom_type"] = "good"
            else:
                inputs["anom_type"] = str(img_file).split("/")[-3]
            if self.is_mask:
                if self.labels_str[index] == NORMAL_PREFIX:
                    mask = Image.new('L', (self.input_res, self.input_res), 0)
                else:
                    mask_file = self.masks[index]
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
        # First load meta file 
        import json
        meta_file = Path(self.meta_dir) / f"{self.category}.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file {meta_file} does not exist.")
        with open(meta_file, 'r') as f:
            meta_data = json.load(f)

        if self.split == "train":
            train_files = meta_data.get('train', None)
            if not train_files:
                raise ValueError(f"No training files found for category {self.category} in meta file.")
            files = [os.path.join(self.data_root, self.category, self.category, file["image_path"]) for file in train_files]
            labels = [file["anomaly_class"] for file in train_files]
            mask_paths = [os.path.join(self.data_root, self.category, self.category, file["mask_path"]) for file in train_files if file["mask_path"]]
        elif self.split == "test":
            test_files = meta_data.get('test', None)
            if not test_files:
                raise ValueError(f"No test files found for category {self.category} in meta file.")
            files = [os.path.join(self.data_root, self.category, self.category, file["image_path"]) for file in test_files]
            labels = [file["anomaly_class"] for file in test_files]
            mask_paths = []
            for f in test_files:
                if f["mask_path"]:
                    mask_paths.append(os.path.join(self.data_root, self.category, self.category, f["mask_path"]))
                else:
                    mask_paths.append(None)
        else:
            raise ValueError(f"Unknown split: {self.split}. Expected 'train' or 'test'.")
        return files, labels, mask_paths
    
if __name__ == "__main__":
    print(len(REALIAD_CLASSES))
    
    data_dir = "/mnt/c/Users/sshun/Downloads/realiad_1024/realiad_1024"
    category = "audiojack"
    input_res = 224
    split = "test"
    meta_dir = "/mnt/c/Users/sshun/Downloads/realiad_jsons/realiad_jsons"
    transform = transforms.Compose([
        transforms.Resize((input_res, input_res), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = RealIAD(
        data_root=data_dir,
        category=category,
        input_res=input_res,
        split=split,
        meta_dir=meta_dir,
        transform=transform,
        is_mask=True,
        cls_label=True,
        anom_only=True,
        normal_only=False,
    )
    print(f"{len(dataset)}")
    
    idx = 801
    sample = dataset[idx]
    org_img = Image.open(sample["filenames"])
    org_img = org_img.convert('RGB')
    org_img = org_img.resize((input_res, input_res), Image.BICUBIC)
    
    # save image
    org_img.save("org_img.jpg")
    
    mask = sample["masks"]
    mask = mask.numpy().astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img.save("mask_img.jpg")
    
    anom_type = sample["anom_type"]
    clsnames = sample["clsnames"]
    clslabels = sample["clslabels"]
    print(f"Anomaly Type: {anom_type}, Class Name: {clsnames}, Class Label: {clslabels}")