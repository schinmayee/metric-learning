#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from matplotlib.mlab import PCA
import matplotlib.pyplot as plt

import argparse
import os
import shutil
import time
import sys

import copy
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
from random import shuffle

import utils

# networks
import model_net

# triplet and loss
import losses

from cub_loader import CUBImages

# Training settings
parser = argparse.ArgumentParser(description='Metric Learning With Triplet Loss and Unknown Classes')

parser.add_argument('--network', type=str, default='Simple',
        help='network architecture to use (default: Simple)')
parser.add_argument('--load', type=str, default='',
                    help='path to checkpoint (default: none)')
parser.add_argument('--output', type=str, default='output',
	            help='output directory (default: output)')
parser.add_argument('--feature-size', type=int, default=64,
        help='size for embeddings/features to learn')

parser.add_argument('--num-train', type=int, default=8,
        help='Number of train classes')
parser.add_argument('--num-val', type=int, default=4,
        help='Number of validation classes')
parser.add_argument('--num-test', type=int, default=4,
        help='Number of test classes')
parser.add_argument('--batch-size', type=int, default=8,
                    help='input batch size for training (default: 8)')

parser.add_argument('--normalize-features', action='store_true', default=False,
                    help='normalize features')

# parameters
feature_size = 0

im_size = 64

use_cmd_split=True  # if false, set the following values to something meaningful
num_train=0
num_val=0
num_test=0
train_classes=None  # triplets_per_class*train_classes should be a multiple of batch size (64 by default)
val_classes=None
test_classes=None
output_dir = ''

# main
def main():
    global args, feature_size, im_size
    global num_train, num_val, num_test
    global train_classes, val_classes, test_classes
    global output_dir

    args = parser.parse_args()

    output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
	    args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # train/val/test split
    if use_cmd_split:
        num_train=args.num_train
        num_val=args.num_val
        num_test=args.num_test
        train_classes=range(num_train)  # triplets_per_class*train_classes should be a multiple of batch size (64 by default)
        val_classes=range(num_train-num_val,num_train)
        test_classes=range(num_train,num_train+num_test)

    # feature size
    feature_size = args.feature_size

    # network
    Net = None
    model = None
    if args.network == 'Simple':
        print('Using simple net')
        Net = model_net.SimpleNet
    elif args.network == 'Inception':
        print('Using inception net')
        Net = model_net.InceptionBased
        # force image size to be 299
        im_size = 299
    elif args.network == 'Squeeze':
        print('Using squeezenet')
        Net = model_net.SqueezeNetBased
        # force image size to be 224
        im_size = 224
    elif args.network == 'Shallow':
        print('Using shallownet')
        Net = model_net.ShallowNet
        # force image size to be 96
        im_size = 96
    elif args.network == 'ResNet':
        print('Using resnet')
        Net = model_net.ResNetBased
        # force image size to be 224
        im_size = 224
    else:
        assert(False)
    model = Net(feature_size=feature_size, im_size=im_size, normalize=args.normalize_features)

    # data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    DLoader = CUBImages
    data_path = os.path.join(dir_path, 'datasets/cub-2011')

    train_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=train_classes, im_size=im_size)
    train_loader = torch.utils.data.DataLoader(
            train_data_set, batch_size=args.batch_size, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(train_data_set))

    val_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=val_classes, im_size=im_size)
    val_loader = torch.utils.data.DataLoader(
            val_data_set, batch_size=args.batch_size/2, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(val_data_set))

    test_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=test_classes, im_size=im_size)
    test_loader = torch.utils.data.DataLoader(
            test_data_set, batch_size=args.batch_size/3, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(test_data_set))

    # resume from a checkpoint
    print("=> loading checkpoint '{}'".format(args.load))
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.load, checkpoint['epoch']))

    model.eval()

    #print('Generating training set embeddings')
    #train_embeddings, train_labels_true = ComputeEmbeddings(train_loader, model)
    print('Generating validation set embeddings')
    val_embeddings, val_labels_true = ComputeEmbeddings(val_loader, model)
    #Print('Generating test set embeddings')
    #Test_embeddings, test_labels_true = ComputeEmbeddings(test_loader, model)

    # select some random classes for visualization
    train_sel = [[0,6,10,20,49,53]]
    val_sel   = [[64,74,80,86,95,98]]
    test_sel  = [[105,129,144,156,183,192]]
    markers = ['o', '^', 'v', 'p', 's', 'D']

    print('Generating PCA for validation set')
    SavePCA(val_embeddings, val_labels_true, val_sel, markers,
	    os.path.join(output_dir, 'val'))

def ComputeEmbeddings(loader, enet):
    global feature_size
    enet.eval()
    embeddings = np.zeros(shape=(len(loader.dataset), feature_size),
                          dtype=float)
    labels_true = np.zeros(shape=(len(loader.dataset)), dtype=int)
    for batch_idx, (data, classes, ids) in enumerate(loader):
        data = Variable(data)

        # compute embeddings
        f = enet(data)
        embeddings[ids.numpy(),:] = f.cpu().data.numpy()
        labels_true[ids.numpy()] = classes.cpu().numpy()
    return embeddings, labels_true

def SavePCA(features, labels, classes, markers, prefix):
    for n, cc in enumerate(classes):
	ids = list()
	cc_ids = list()
	for c in cc:
	    c_ids = list(np.where(labels == c)[0])
	    #c_ids = np.random.choice(c_ids, 20)  # select random 20
	    ids = ids + list(c_ids)
	    cc_ids.append(list(c_ids))
	ids = np.array(ids)
	samples = features[ids,:]
	pca = PCA(samples)
	num_plotted = 0
	for i in range(len(cc)):
	    num = len(cc_ids[i])
	    plt.plot(pca.Y[num_plotted:num_plotted+num,0],
		     pca.Y[num_plotted:num_plotted+num,1],
		     markers[i], markersize=5, alpha=0.5)
	    num_plotted += num
	plt.axis('off')
	plt.savefig(prefix + ('_%d.png' % n))

if __name__ == '__main__':
    main()  
