'''
MVTec AD 2 Dataset.
The dataset class provides functions to load images (and ground truth if
applicable) from the MVTec AD 2 dataset and to store anomaly images in the
correct structure for evaluating performance on the evaluation server.
'''

# Copyright (C) 2025 MVTec Software GmbH
# SPDX-License-Identifier: CC-BY-NC-4.0

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

AD2_CLASSES = [
    'can',
    'fabric',
    'fruit_jelly',
    'rice',
    'sheet_metal',
    'vial',
    'wallplugs',
    'walnuts',
]

class MVTecAD2(Dataset):
    """Dataset class for MVTec AD 2 objects.

    Args:
        mad2_object (str): can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
        split (str): train, validation, test_public, test_private, test_private_mixed
        transform (function, optional): transform applied to samples, defaults to 'to_tensor'
    """

    def __init__(
        self,
        root: str,
        category,
        split,
        transform=to_tensor,
        **kwargs,
    ):
        assert split in {
            'train',
            'validation',
            'test_public',
            'test_private',
            'test_private_mixed',
        }, f'unknown split: {split}'

        assert (
            category in MVTEC_AD_2_OBJECTS
        ), f'unknown MVTec AD 2 object: {category}'

        self.object = category
        self.split = split
        self.transform = transform

        self._image_base_dir = root

        self._object_dir = os.path.join(self._image_base_dir, category)
        # get all images from the split
        self._image_paths = sorted(glob.glob(self._get_pattern()))

    def _get_pattern(self) -> str:
        if 'private' in self.split:
            return os.path.join(
                self._object_dir, self.split, '[0-9][0-9][0-9]*.png'
            )
        return os.path.join(
            self._object_dir,
            self.split,
            '[gb][oa][od]*',
            '[0-9][0-9][0-9]*.png',
        )

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Get dataset item for the index ``idx``.

        Args:
            idx (int): Index to get the item.

        Returns:
            dict[str,  str | torch.Tensor]: Dict containing the sample image,
            image path, and the relative anomaly image output path for both
            image types continuous and thresholded.
        """

        image_path = self._image_paths[idx]
        sample = default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        cls_name = str(image_path).split("/")[-4]
        cls_label = MVTEC_AD_2_OBJECTS.index(cls_name)

        return {
            'samples': sample,
            'filenames': image_path,
            'is_anomaly': ("good" not in str(image_path)),
            'clsnames': cls_name,
            'clslabels': cls_label,
            'rel_out_path_cont': self.get_relative_anomaly_image_out_path(idx),
            'rel_out_path_thresh': self.get_relative_anomaly_image_out_path(
                idx, True
            ),
        }

    @property
    def image_paths(self):
        return self._image_paths

    @property
    def has_segmentation_gt(self) -> bool:
        return self.split == 'test_public'

    def get_relative_anomaly_image_out_path(self, idx, thresholded=False):
        """Returns a path relative to the experiment directory
        for storing the (thresholded) anomaly image in the required structure.

        Args:
            idx (int): sample index
            thresholded (bool): return output path for thresholded image,
            defaults to 'False'

        Returns:
            str: relative output path to write anomaly image
        """

        image_path = Path(self._image_paths[idx])
        relpath = image_path.relative_to(self._image_base_dir)

        if not thresholded:
            base_dir = 'anomaly_images'
            suffix = '.tiff'
        else:
            base_dir = 'anomaly_images_thresholded'
            suffix = '.png'

        return os.path.join(base_dir, relpath.with_suffix(suffix))

    def get_gt_image(self, idx):
        """Returns the ground truth image where values of 255 denote
        anomalous pixels and values of 0 anomaly-free ones. For good images
        'None' is returned.
        In case no segmentation ground truth is available
        (test_private/test_private_mixed) 'None' is returned as well.

        Args:
            idx (int): sample index

        Returns:
            numpy.array or None: ground truth image if available
        """
        gt_image = None
        if (
            self.has_segmentation_gt
            and 'good' not in self.get_relative_anomaly_image_out_path(idx)
        ):
            image_path = self.image_paths[idx]
            base_path, file_name = image_path.split('/bad/')
            gt_image_path = os.path.join(
                base_path, 'ground_truth/bad', file_name
            ).replace('.png', '_mask.png')

            gt_image_pil = Image.open(gt_image_path)
            gt_image = np.asarray(gt_image_pil)

        return gt_image
