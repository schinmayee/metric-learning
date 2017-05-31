from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from random import shuffle

"""
Base class for sampling.
"""
class TripletSampler(object):
    def __init__(self, num_classes, num_samples):
        self.num_classes = int(num_classes)
        self.num_samples = int(num_samples)

    """
    Reset all samples, that is, clear all mined hard examples.
    """
    def Reset(self):
        pass

    """
    Call this after every batch to generate a list of hard negatives.
    """
    def SampleNegatives(self, dista, distb, triplet_loss, ids):
        print("Implement me!!")
        pass

    """
    Call this when regenerating list of triplets, to get a set of negative pairs.
    """
    # TODO: may want to update the signature.
    def ChooseNegatives(self, num):
        print("Implement me!!")
        pass

"""
Get N hardest:
Every time SampleNegatives is invoked, this chooses the
triplets with N hardest negatives from a set of already constructred triplets,
such as the triplets used in training/
When ChooseNegatives is invoked, this returns a set of hard negatives from
triplets that it has seen before.
"""
class NHardestTripletSampler(TripletSampler):
    def __init__(self, num_classes, num_samples):
        super(NHardestTripletSampler, self).__init__(num_classes, num_samples)
        self.negatives = []
        self.dist_neg = []

    def Reset(self):
        self.negatives = []
        self.dist_neg = []

    """
    Negatives with least distb.
    """
    def SampleNegatives(self, dista, distb, triplet_loss, ids):
        distb = distb.cpu()
        assert(self.num_samples <= dista.size()[0])
        idx1, idx2, idx3 = ids
        # sort by distance between anchor and negative
        sortd, indices = torch.sort(distb, descending=False, dim=0)
        sel_indices = indices[0:self.num_samples].data.numpy().reshape((self.num_samples))
        anchor = idx1.numpy()[sel_indices].reshape((self.num_samples))
        negs   = idx3.numpy()[sel_indices].reshape((self.num_samples))
        self.negatives += zip(anchor, negs)
        self.dist_neg += list(sortd.data.numpy()[sel_indices].reshape((self.num_samples)))

    """
    Now get some triplets for regenerating triplets.
    """
    def ChooseNegatives(self, num):
        l = len(self.negatives)
        assert(l >= num)
        # sort by distance between anchor and negative
        sorted_indices = np.argsort(self.dist_neg)
        sel_indices = sorted_indices[0:num]
        return ([self.negatives[i] for i in sel_indices])

"""
Semihard sampler -- selects examples where distance between anchor and negative
is less than the distance between anchor and positive.
"""
class SemiHardTripletSampler(TripletSampler):
    def __init__(self, num_classes, num_samples):
        super(SemiHardTripletSampler, self).__init__(num_classes, num_samples)
        self.negatives = []

    def Reset(self):
        self.negatives = []

    """
    Negatives with distb < dista.
    """
    def SampleNegatives(self, dista, distb, triplet_loss, ids):
        dista = dista.cpu()
        distb = distb.cpu()
        assert(self.num_samples <= dista.size()[0])
        idx1, idx2, idx3 = ids
        # select examples where distance to negative is less than distance to positive
        sel_indices = np.where(distb.data.numpy() < dista.data.numpy())[0]
        sel_indices = np.random.choice(sel_indices, self.num_samples)
        anchor = idx1.numpy()[sel_indices].reshape((self.num_samples))
        negs   = idx3.numpy()[sel_indices].reshape((self.num_samples))
        self.negatives += zip(anchor, negs)

    """
    Now get some triplets for regenerating triplets.
    """
    def ChooseNegatives(self, num):
        sel_indices = np.random.choice(range(len(self.negatives)), num)
        return ([self.negatives[i] for i in sel_indices])

"""
Classification based sampler.
"""
class ClassificationBasedSampler(object):
    def __init__(self, num_classes, num_samples):
        self.num_classes = int(num_classes)
        self.num_samples = int(num_samples)
        self.negatives = []

    def Reset(self):
        self.negatives = []

    def SampleNegatives(self, labels_true, labels_pred):
        neg_indices = np.where(labels_true != labels_pred)[0]
        true_classes = labels_true[neg_indices]
        pred_classes = labels_pred[neg_indices]  # what cluster does this point falsely belong to?

        cor_indices = np.where(labels_true == labels_pred)[0]  # some points correctly clustered
        cor_classes = labels_true[cor_indices]
        anchor_candidates = dict()
        for c in np.unique(cor_classes):
            subset_indices = np.where(cor_classes == c)[0]
            anchor_candidates[c] = cor_indices[subset_indices]

        # now get an anchor point (a correct point correctly there in that cluster)
        for i in np.random.permutation(len(neg_indices)):
            pred_class = pred_classes[i]  # predicted class for incorrectly classified point
            if pred_class in anchor_candidates.keys():  # there is an anchor point, a point correctly sent to that class
                self.negatives.append((np.random.choice(anchor_candidates[pred_class]), neg_indices[i]))
            if len(self.negatives) == self.num_samples:
                break

    def ChooseNegatives(self, num):
        sel_indices = np.random.choice(range(len(self.negatives)), num)
        return ([self.negatives[i] for i in sel_indices])
