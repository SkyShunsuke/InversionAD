

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from .mvtec_ad import MVTecAD, AD_CLASSES
from .visa import VisA, VISA_CLASSES
from .mpdd import MPDD, MPDD_CLASSES

import random

def build_transforms(img_size, transform_type):
    # standarization
    default_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    if transform_type == 'default':
        return default_transform
    elif transform_type == 'imagenet':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif transform_type == 'crop':
        return transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif transform_type == 'rotate':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    elif transform_type == 'ddad':
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
    else:
        raise ValueError(f"Invalid transform: {transform_type}")

def build_dataset(*, dataset_name: str, data_root: str, train: bool, img_size: int, transform_type: str, **kwargs):
    if dataset_name == 'mvtec_ad':
        return MVTecAD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    elif dataset_name == 'visa':
        return VisA(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, **kwargs)
    elif dataset_name == 'mpdd':
        return MPDD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    elif dataset_name == 'mvtec_ad_all':
        dss = []
        for cat in AD_CLASSES:
            kwargs['category'] = cat
            dss.append(MVTecAD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs))
        return ConcatDataset(dss)
    elif dataset_name == 'visa_all':
        dss = []
        for cat in VISA_CLASSES:
            kwargs['category'] = cat
            dss.append(VisA(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs))
        return ConcatDataset(dss)
    elif dataset_name == 'mpdd_all':
        dss = []
        for cat in MPDD_CLASSES:
            kwargs['category'] = cat
            dss.append(MPDD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs))
        return ConcatDataset(dss)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")