"""
The Caltech-UCSD birds dataset.
"""

import os
import math

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd 
from PIL import Image
import hard_mining


def default_image_loader(path):
    return Image.open(path).convert('RGB')

class CUB_t(data.Dataset):
    def __init__(self, root, n_triplets=10000, num_classes=-1, train=True, transform=None, im_size=64):

        self.loader = default_image_loader
        
        self.transform = transform
        self.train = train  # training set or test set
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

        split = [int(line.split()[1]) for line in
                    open(os.path.join(root, 'train_test_split.txt'),'r')]

        if train:
            target = 1
        else:
            target = 0

        # load list and metadata for train/test set
        # paths
        self.images = [image for image, val in zip(images, split) if val == target]
        # labels
        self.labels = np.array([label for label, val in zip(labels, split) if val == target])
        # boxes
        self.boxes  = np.array([box for box, val in zip(boxes, split) if val == target])

        if num_classes < 0:
            self.num_classes = len(labels)
        else:
            self.num_classes = min(num_classes, len(labels))
        print("CUB triplet loader initialized for %d classes, %d triplets" % (self.num_classes, n_triplets))

        # make triplets
        self.num_triplets = n_triplets
        self.make_triplet_list(n_triplets)


    def __getitem__(self, index):
        idx1, idx2, idx3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.im_base_path, self.images[idx1]))
        img2 = self.loader(os.path.join(self.im_base_path, self.images[idx2]))
        img3 = self.loader(os.path.join(self.im_base_path, self.images[idx3]))
        b1, b2, b3 = self.boxes[idx1], self.boxes[idx2], self.boxes[idx3]
        img1 = img1.crop(b1)
        img2 = img2.crop(b2)
        img3 = img3.crop(b3)
        img1 = img1.resize((self.im_size, self.im_size))
        img2 = img2.resize((self.im_size, self.im_size))
        img3 = img3.resize((self.im_size, self.im_size))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, idx1, idx2, idx3

    def __len__(self):
        return self.num_triplets

    def make_triplet_list(self, ntriplets):
        print('Processing Triplet Generation ...')
        self.triplets = []
        nc = int(self.num_classes)
        for class_idx in range(1,nc+1):
            # a, b, c are index of labels where it's equal to class_idx
            a = np.random.choice(np.where(self.labels==class_idx)[0],
                                 int(ntriplets/nc), replace=True)
            b = np.random.choice(np.where(self.labels==class_idx)[0],
                                 int(ntriplets/nc), replace=True)
            while np.any((a-b)==0): #aligning check
                np.random.shuffle(b)
            c = np.random.choice(np.where(self.labels!=class_idx)[0],
                                 int(ntriplets/nc), replace=True)

            for i in range(a.shape[0]):
                self.triplets.append((a[i], b[i], c[i]))

        print('Done!')

    def regenerate_triplet_list(self, sampler, frac_hard):
        assert(self.train)
        print("Processing Triplet Regeneration ...")
        # negatives is a tuple of anchors and negative examples
        num_random_triplets = self.num_triplets*(1.0-frac_hard)
        # adjust number of random triplets so that it is a multiple of num_classes
        num_random_triplets = int(math.ceil(num_random_triplets)/self.num_classes)*self.num_classes
        num_hard = self.num_triplets - num_random_triplets
        print("Number of hard triplets %d ..." % num_hard)
        self.make_triplet_list(num_random_triplets)
        neg_hard_triplets = sampler.ChooseNegatives(num_hard)
        self.triplets += neg_hard_triplets
        np.random.shuffle(self.triplets)
