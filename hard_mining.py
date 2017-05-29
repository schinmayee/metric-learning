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
from visdom import Visdom
import numpy as np
from random import shuffle

"""
Base class for sampling.
"""
class Sampler(object):
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
    Call this after every batch to generate a list of hard positives.
    """
    def SamplePositives(self, dista, distb, triplet_loss, ids):
        print("Implement me!!")
        pass

    """
    Call this when regenerating list of triplets, to get a set of negative pairs,
    or triplets with negative examples.
    """
    # TODO: may want to update the signature.
    def ChooseNegatives(self, num):
        print("Implement me!!")
        pass

    """
    Call this when regenerating list of triplets, to get a set of positive pairs,
    or triplets with positive examples.
    """
    # TODO: may want to update the signature.
    def ChoosePositives(self, num):
        print("Implement me!!")
        pass

"""
Get N hardest:
Every time SampleNegatives or SamplePositives is invoked, this chooses the
triplets with N hardest positives and N hardest negatives.
When ChooseNegatives or ChoosePositives is invoked, this returns a random
set of triplets from the sampled triplets.
"""
class NHardestTripletSampler(Sampler):
    def __init__(self, num_classes, num_samples):
        super(NHardestTripletSampler, self).__init__(num_classes, num_samples)
        self.negatives = []
        self.positives = []

    def Reset(self):
        self.negatives = []
        self.positives = []

    """
    Negatives with least distb.
    """
    def SampleNegatives(self, dista, distb, triplet_loss, ids):
        idx1, idx2, idx3 = ids
        sortd, indices = torch.sort(distb, descending=False, dim=0)
        sel_indices = indices[0:self.num_samples].data.numpy().reshape((self.num_samples))
        anchor = idx1.numpy()[sel_indices].reshape((self.num_samples))
        pos    = idx2.numpy()[sel_indices].reshape((self.num_samples))
        negs   = idx3.numpy()[sel_indices].reshape((self.num_samples))
        self.negatives += zip(anchor, pos, negs)

    """
    Positives with max dista.
    """
    def SamplePositives(self, dista, distb, triplet_loss, ids):
        idx1, idx2, idx3 = ids
        sortd, indices = torch.sort(dista, descending=True, dim=0)
        sel_indices = indices[0:self.num_samples].data.numpy().reshape((self.num_samples))
        anchor = idx1.numpy()[sel_indices].reshape((self.num_samples))
        pos    = idx2.numpy()[sel_indices].reshape((self.num_samples))
        negs   = idx3.numpy()[sel_indices].reshape((self.num_samples))
        self.positives += zip(anchor, pos, negs)

    """
    Now get some triplets for regenerating triplets.
    """
    def ChooseNegatives(self, num):
        l = len(self.negatives)
        # we could sort here but I am just going to select randomly -- chinmayee
        indices = np.random.choice(l, num, replace=True)
        return ([self.negatives[i] for i in indices])

    """
    Now get some triplets for regenerating triplets.
    """
    def ChoosePositives(self, num):
        l = len(self.positives)
        # we could sort here but I am just going to select randomly -- chinmayee
        indices = np.random.choice(l, num, replace=True)
        return ([self.positives[i] for i in indices])
