"""
The Caltech-UCSD birds dataset.
"""

import os
import math

import numpy as np

import torch
import torch.utils.data as data

from PIL import Image

import hard_mining


def default_image_loader(path):
    return Image.open(path).convert('RGB')

class CUBImages(data.Dataset):
    def __init__(self, root, classes=range(200), transform=None, im_size=128):

        self.loader = default_image_loader
        
        self.transform = transform
        self.im_size = im_size

        # paths
        self.root = root
        self.im_base_path = os.path.join(root, 'images')

        # load metadata
        images = [line.split()[1] for line in
                    open(os.path.join(root, 'images.txt'), 'r')]
        labels = [int(line.split()[1]) - 1 for line in 
                    open(os.path.join(root, 'image_class_labels.txt'), 'r')]
        birdnames = [line.split()[1] for line in
                      open(os.path.join(root, 'classes.txt'), 'r')]
        boxes = [[int(round(float(c))) for c in line.split()[1:]] for line in
                 open(os.path.join(root, 'bounding_boxes.txt'),'r')]
        name_to_id = dict(zip(birdnames, range(len(birdnames))))

        # which classes to include
        self.classes = classes
        self.num_classes = len(classes)
        split = [l in classes for l in labels]

        # load list and metadata for train/test set
        # paths
        self.images = [image for image, val in zip(images, split) if val]
        # labels
        self.labels = np.array([label for label, val in zip(labels, split) if val])
        # boxes
        self.boxes  = np.array([box for box, val in zip(boxes, split) if val])
        # number of images
        self.num_images = len(self.images)

        print("CUB loader initialized for %d classes, %d images" % (self.num_classes, self.num_images))

    def __len__(self, index):
        return self.num_images

    def __getitem__(self, index):
        img = self.loader(os.path.join(self.im_base_path, self.images[index]))
        img.crop(self.boxes[index])
        img = img.resize(self.im_size, self.im_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, index
