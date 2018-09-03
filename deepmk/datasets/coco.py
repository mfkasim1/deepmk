import os
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets

__all__ = ["CocoDetection"]

class CocoDetection(datasets.CocoDetection):
    """
    Modification from torchvision.datasets.CocoDetection where the default
    target transformation here is to create multiple channel annotations.
    The image is a PIL image of the data.
    The target is a torch array with the same (h x w) size with the image.
    """
    def __init__(self, root, ann_file, transform=None, target_transform=None):
        # get the self.coco object from the parent object
        super(CocoDetection, self).__init__(root, ann_file, transform, None)
        self.target_transform2 = target_transform

        # get the categories
        self.category_ids = sorted(self.coco.cats.keys())
        self.ncategories = len(self.category_ids)
        self.cat2idx = {self.category_ids[i]: i for i in range(self.ncategories)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target).
                image is a PIL image
                target is multiple channel of the annotations.
        """
        img, target = super(CocoDetection, self).__getitem__(index)

        # convert the annotations to mask
        shape = tuple([self.ncategories] + list(img.size[::-1]))
        ann_img = np.zeros(shape)
        for ann in target:
            idx = self.cat2idx[ann['category_id']]
            ann_img[idx,:,:] = self.coco.annToMask(ann)

        # convert to torch array
        target = torch.from_numpy(ann_img)

        # apply target transformations
        if self.target_transform2 is not None:
            target = self.target_transform2(target)

        return img, target
