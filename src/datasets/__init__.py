

from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

from .mnist import MNISTWrapper
from .mvtec_ad import MVTecAD, AD_CLASSES
from .mvtec_loco import MVTecLOCO, LOCO_CLASSES
from .visa import VisA, VISA_CLASSES

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
    if dataset_name == 'mnist':
        return MNISTWrapper(root=data_root, train=train, transform=build_transforms(img_size, transform_type))
    elif dataset_name == 'mvtec_ad':
        return MVTecAD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    elif dataset_name == 'visa':
        return VisA(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, **kwargs)
    elif dataset_name == 'mvtec_loco':
        return MVTecLOCO(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
            transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs)
    elif dataset_name == 'mvtec_ad_mc':
        dss = []
        for cat in AD_CLASSES:
            kwargs['category'] = cat
            dss.append(MVTecAD(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), is_mask=True, cls_label=True, **kwargs))
        return ConcatDataset(dss)
    elif dataset_name == 'visa_mc':
        dss = []
        for cat in VISA_CLASSES:
            kwargs['category'] = cat
            dss.append(VisA(data_root=data_root, input_res=img_size, split='train' if train else 'test', \
                transform=build_transforms(img_size, transform_type), **kwargs))
        return ConcatDataset(dss)
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    
class ICLDataLoader:
    """
    ICL DataLoader collate batch from one class dataset, and iterate over all classes. 
    """
    def __init__(self, dataset_list, collate_fn, batch_size):
        self.dataset_list = dataset_list
        self.collate_fn = collate_fn
        self.num_classes = len(dataset_list)
        self.batch_size = self.batch_size
        
        self.dataloader_list = [iter(DataLoader(ds, batch_size, shuffle=True, collate_fn=collate_fn)) for ds in dataset_list]

    def __next__(self):
        # choose a random class
        class_idx = random.randint(0, self.num_classes - 1)
        try:
            data = next(self.dataloader_list[class_idx])
        except StopIteration:
            self.dataloader_list[class_idx] = iter(DataLoader(self.dataset_list[class_idx], self.batch_size, shuffle=True, collate_fn=self.collate_fn))
            data = next(self.dataloader_list[class_idx])
        return data
    
    def __iter__(self):
        return self            
        

class EvalDataLoader:
    def __init__(self, dataset, num_repeat, batch_size):
        self.dataset = dataset
        self.num_repeat = num_repeat
        self.batch_size = batch_size
        # We repeat each sample in the dataset for the number of times it is required to be evaluated.
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)  
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = next(self.iterator)  
        
        imgs = data["samples"] # (B, C, H, W) -> (C, H, W)
        labels = data["labels"]
        fnames = data["filenames"]
        clsnames = data["clsnames"]
        anomtypes = data["anom_type"]
        clslabels = data["clslabels"]
        gtmasks = data["masks"]  # gt masks
        
        imgs = imgs.repeat_interleave(self.num_repeat, dim=0)  # (B*N, C, H, W)
        labels = labels.repeat_interleave(self.num_repeat, dim=0)
        fnames = [fname for fname in fnames for _ in range(self.num_repeat)]
        clsnames = [clsname for clsname in clsnames for _ in range(self.num_repeat)]
        anomtypes = [anomtype for anomtype in anomtypes for _ in range(self.num_repeat)]
        clslabels = clslabels.repeat_interleave(self.num_repeat, dim=0)
        gtmasks = gtmasks.repeat_interleave(self.num_repeat, dim=0)

        # use shared masks accross all test samples
        collated_batch = {
            "samples": imgs,
            "labels": labels,
            "filenames": fnames,
            "clsnames": clsnames,
            "anom_type": anomtypes,
            "clslabels": clslabels,
            "masks": gtmasks,
        }
        return collated_batch
    
    def __len__(self):
        return len(self.dataloader)