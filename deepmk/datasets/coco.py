import os
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

__all__ = ["CocoDetection"]

class CocoDetection(datasets.CocoDetection):
    """
    Modification from torchvision.datasets.CocoDetection where the default
    target transformation here is to create multiple channel annotations.
    The image is a torch array with shape (c x h x w), where typically c=3.
    The target is a torch array with the same (c1 x h x w) size with the image,
    but with channels c1 equals the number of classes.
    """
    def __init__(self, root, ann_file, both_transform=None,
            img_transform=None, target_transform=None):
        # get the self.coco object from the parent object
        super(CocoDetection, self).__init__(root, ann_file, None, None)
        self.both_transform = both_transform
        self.img_transform = img_transform
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
                image is a torch array of the image.
                target is multiple channel of the annotations.
                Both are FloatTensor.
        """
        img, target = super(CocoDetection, self).__getitem__(index)

        # convert the annotations to mask
        shape = tuple([self.ncategories] + list(img.size[::-1]))
        ann_img = np.zeros(shape)
        for ann in target:
            idx = self.cat2idx[ann['category_id']]
            ann_img[idx,:,:] += self.coco.annToMask(ann)
        ann_img = (ann_img > 0) * 1.0

        # convert to torch array
        img = transforms.ToTensor()(img)
        target = torch.FloatTensor(ann_img)

        # apply image and target transformations
        if self.both_transform is not None:
            stack_img_target = torch.cat((img, target), dim=0) # stack on the channels
            stack_img_target = self.both_transform(stack_img_target)
            img = stack_img_target[:img.shape[0],:,:]
            target = stack_img_target[img.shape[0]:,:,:]
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.target_transform2 is not None:
            target = self.target_transform2(target)

        return img, target
