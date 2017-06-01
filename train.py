#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import shutil
import time
import sys

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import numpy as np
from random import shuffle

import pickle

import utils

# networks
import model_net

# triplet and loss
import triplet_net
import losses

# sampling
import hard_mining

# data loader
from triplet_cub_loader import CUBTriplets
from cub_loader import CUBImages

# sklearn for clustering and evaluating clusters
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

# Training settings
parser = argparse.ArgumentParser(description='Metric Learning With Triplet Loss and Unknown Classes')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='adam beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='adam beta2 (default: 0.999)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--reg', type=float, default=1e-3,
                    help='regularization for embedding (default: 1e-3)')
parser.add_argument('--resume', type=str, default='',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--loss', type=str, default='HingeL2',
        help='loss mechanism (default: HingeL2)')
parser.add_argument('--data', type=str, default='cub-2011',
        help='dataset (default: cub-2011)')

parser.add_argument('--triplet-freq', type=int, default=10,
                    help='epochs before new triplets list (default: 10)')
parser.add_argument('--val-freq', type=int, default=2,
        help='epochs before validating on validation set (default: 2)')
parser.add_argument('--results-freq', type=int, default=2,
        help='epochs before saving results (default: 2)')
parser.add_argument('--test-results-freq', type=int, default=10,
        help='epochs before saving results (default: 10)')

parser.add_argument('--network', type=str, default='Simple',
        help='network architecture to use (default: Simple)')
parser.add_argument('--log-interval', type=int, default=2,
        help='how many batches to wait before logging training status (default: 2)')
parser.add_argument('--feature-size', type=int, default=64,
        help='size for embeddings/features to learn')

parser.add_argument('--num-train', type=int, default=4,
        help='Number of train classes')
parser.add_argument('--num-val', type=int, default=4,
        help='Number of validation classes')
parser.add_argument('--num-test', type=int, default=2,
        help='Number of test classes')
parser.add_argument('--triplets-per-class', type=int, default=16,
        help='Number of triplets per class')

parser.add_argument('--normalize-features', action='store_true', default=False,
                    help='normalize features')
parser.add_argument('--in-triplet-hard', action='store_true', default=False,
                    help='enables in triplet hard mining')
parser.add_argument('--mining', type=str, default='Hardest',
        help='Method to use for mining hard examples')

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
triplets_per_class=0  # keep at least 16 triplets per class, later increase to 32/64

hard_frac = 0.5

# globals
best_acc = 0
best_precision = 0
best_recall = 0
best_f1 = 0
best_model = None

runs_dir = ''

epochs = list()
train_losses = list()
val_losses = list()
triplet_accs = list()
classification_accs = list()

sampler = None

# main
def main():
    global args, feature_size, im_size
    global best_acc, best_precision, best_recall, best_f1, best_model
    global epochs, train_losses, val_losses
    global triplet_accs, classification_accs
    global runs_dir
    global num_train, num_val, num_test, triplets_per_class, use_cmd_split
    global train_classes, val_classes, test_classes
    global Sampler

    args = parser.parse_args()
    
    runs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            ('runs/r-%s-%s-f%d-%s' %
                (args.network, args.loss, args.feature_size, time.strftime('%m-%d-%H-%M'))))
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    command = ' '.join(sys.argv)
    with open(os.path.join(runs_dir, 'command.sh'), 'w') as c:
        c.write(command)
        c.write('\n')

    # train/val/test split
    if use_cmd_split:
        num_train=args.num_train
        num_val=args.num_val
        num_test=args.num_test
        train_classes=range(num_train)  # triplets_per_class*train_classes should be a multiple of batch size (64 by default)
        val_classes=range(num_train,num_train+num_val)
        test_classes=range(num_train+num_val,num_train+num_val+num_test)
        triplets_per_class=args.triplets_per_class
    assert(triplets_per_class*len(train_classes)%args.batch_size == 0)

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

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
    else:
        assert(False)
    model = Net(feature_size=feature_size, im_size=im_size, normalize=args.normalize_features)

    # triplet network
    tnet = triplet_net.TripletNet(model)

    if args.cuda:
        tnet.cuda()

    # loss to use
    if args.loss == 'HingeL2':
        criterion = losses.SimpleHingeLoss
    elif args.loss == 'SquareHingeL2':
        criterion = losses.SimpleSquareHingeLoss
    elif args.loss == 'Ratio':
        criterion = losses.RatioLoss
    else:
        assert(False)

    # sampler to use
    if args.mining == 'Hardest':
        sampler = hard_mining.NHardestTripletSampler(
                len(train_classes),
                int((hard_frac+hard_frac/2)*args.batch_size))
    elif args.mining == 'SemiHard':
        sampler = hard_mining.SemiHardTripletSampler(
                len(train_classes),
                int((hard_frac+hard_frac/2)*args.batch_size))
    elif args.mining == 'KMeans':
        sampler = hard_mining.ClassificationBasedSampler(
                len(train_classes),
                int((hard_frac+hard_frac/2)*len(train_classes)*triplets_per_class)
                )
    else:
        assert(False)
    

    # data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.data == 'cub-2011':
        TLoader = CUBTriplets
        DLoader = CUBImages
        data_path = os.path.join(dir_path, 'datasets/cub-2011')
    else:
        assert(False)

    train_data_set_t = TLoader(data_path,
                             n_triplets=triplets_per_class*len(train_classes),
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                             ]),
                             classes=train_classes, im_size=im_size)
    train_loader_t = torch.utils.data.DataLoader(
        train_data_set_t, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=train_classes, im_size=im_size)
    train_loader = torch.utils.data.DataLoader(
            train_data_set, batch_size=args.batch_size, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(train_data_set),
            **kwargs)

    val_data_set_t = TLoader(data_path,
                              n_triplets=triplets_per_class*len(val_classes),
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                              ]),
                              classes=val_classes, im_size=im_size)
    val_loader_t = torch.utils.data.DataLoader(
        val_data_set_t, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=val_classes, im_size=im_size)
    val_loader = torch.utils.data.DataLoader(
            val_data_set, batch_size=args.batch_size, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(val_data_set),
            **kwargs)

    test_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=test_classes, im_size=im_size)
    test_loader = torch.utils.data.DataLoader(
            test_data_set, batch_size=args.batch_size, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(test_data_set),
            **kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    net_params = tnet.SetLearningRate(args.lr*0.1, args.lr)
    optimizer = optim.Adam(net_params, lr=args.lr,
                           betas=[args.beta1,args.beta2])

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    labels_true = None
    labels_predicted = None

    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train_loss = Train(train_loader_t, tnet, criterion, optimizer, epoch, sampler)

        # evaluate on validation set
        if epoch % args.val_freq == 0:
            val_loss, triplet_acc = TestTriplets(val_loader_t, tnet, criterion)
            val_results = ComputeClusters(val_loader, model, len(val_classes))
            acc = val_results['accuracy']
            precision = val_results['precision']
            recall = val_results['recall']
            f1 = val_results['f1']
            labels_true = val_results['true']
            labels_predicted = val_results['predicted']

            # remember best acc and save checkpoint
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                best_model = copy.deepcopy(model)

            
            print('Best cluster: Accuracy %f, Precision %f, Recall %f, F1 %f\n' % (
                best_acc, best_precision, best_recall, best_f1))
            SaveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': tnet.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

            # save data for 2 plots here:
            #   1. train and test loss (triplet)
            #   2. triplet and cluster based classification accuracy on validation set
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            triplet_accs.append(triplet_acc)
            classification_accs.append(acc)
            

        # reset sampler and regenerate triplets every few epochs
        if epoch % args.triplet_freq == 0:
            if args.mining == 'KMeans':
                print('Generating cluster classification on training data ...')
                train_results = ComputeClusters(train_loader, model, len(val_classes))
                labels_true = train_results['true']
                labels_pred = train_results['predicted']
                sampler.SampleNegatives(labels_true, labels_pred)
            train_data_set_t.regenerate_triplet_list(sampler, hard_frac)
            # then reset sampler
            sampler.Reset()

        # save final results and plot loss/accuracy with training
        if epoch % args.results_freq == 0:
            print('Results are being saved to %s' % runs_dir)
            utils.SavePlots(runs_dir, epochs, train_losses, val_losses,
                            triplet_accs, classification_accs)
            pickle.dump(epochs, open(os.path.join(runs_dir, 'epochs'), 'w'))
            pickle.dump(train_losses, open(os.path.join(runs_dir, 'train_losses'), 'w'))
            pickle.dump(val_losses, open(os.path.join(runs_dir, 'val_losses'), 'w'))
            pickle.dump(triplet_accs, open(os.path.join(runs_dir, 'triplet_accs'), 'w'))
            pickle.dump(classification_accs, open(os.path.join(runs_dir, 'classification_accs'), 'w'))

            # at the end, save some query results for visualization
            val_results = ComputeClusters(val_loader, best_model, len(val_classes))
            SaveClusterResults(runs_dir, 'val', val_results, val_data_set)

        if epoch % args.test_results_freq == 0:
            # also run the model and kmeans over test data and save the results
            # over test data, BUT DO NOT use this for tuning hyper-parameters
            print('Saving test results!!')
            test_results = ComputeClusters(test_loader, best_model, len(test_classes))
            SaveClusterResults(runs_dir, 'test', test_results, test_data_set)
            

def Train(train_loader_t, tnet, criterion, optimizer, epoch, sampler):
    losses = AverageMeter()
    loss_accs = AverageMeter()
    emb_norms = AverageMeter()
    
    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3, idx1, idx2, idx3) in enumerate(train_loader_t):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, distc, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        # forward pass
        loss_triplet = criterion(dista, distb, distc, target, args.margin, args.in_triplet_hard)

        if args.mining == 'Hardest' or args.mining == 'SemiHard':
            sampler.SampleNegatives(dista, distb, loss_triplet, (idx1, idx2, idx3))
        
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + args.reg * loss_embedd

        # measure loss accuracy and record loss
        loss_acc = LossAccuracy(dista, distb, distc, args.margin)
        losses.update(loss_triplet.data[0], data1.size(0))
        loss_accs.update(loss_acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Loss Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader_t.dataset),
                losses.val, losses.avg, 
                100. * loss_accs.val, 100. * loss_accs.avg,
                emb_norms.val, emb_norms.avg))

    return loss_accs.avg

def ComputeClusters(test_loader, enet, num_clusters):
    global feature_size
    enet.eval()
    embeddings = np.zeros(shape=(len(test_loader.dataset), feature_size),
                          dtype=float)
    labels_true = np.zeros(shape=(len(test_loader.dataset)), dtype=int)
    for batch_idx, (data, classes, ids) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        # compute embeddings
        f = enet(data)
        embeddings[ids.numpy(),:] = f.cpu().data.numpy()
        labels_true[ids.numpy()] = classes.cpu().numpy()
    print('Generated embeddings, now running k-means for %d clusters...' % num_clusters)

    # initialize centroid  for each cluster
    unique_classes = np.unique(labels_true)
    num_classes = len(unique_classes)
    initial_centers = np.zeros(shape=(num_clusters, feature_size), dtype=float)
    for i in range(num_classes):
        c_ids = np.where(labels_true == unique_classes[i])
        use_im = np.random.choice(c_ids[0])
        initial_centers[i,:] = embeddings[use_im,:]

    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1,
                          max_iter=1000, tol=1e-3,
                          init=initial_centers, n_init=1)
    labels_predicted = kmeans_model.fit_predict(embeddings)

    # map predicted clusters to actual class ids
    cluster_to_class = np.zeros(shape=(num_classes,), dtype=int)
    for i in range(num_classes):
        # figure out which class this cluster must be
        # set of points that belong to this cluster
        cluster_points = np.where(labels_predicted == i)
        # true class labels for these points
        actual_labels = labels_true[cluster_points]
        # map cluster to most frequently occuring class
        unique, indices = np.unique(actual_labels, return_inverse=True)
        mode = unique[np.argmax(
            np.apply_along_axis(
                np.bincount, 0, indices.reshape(actual_labels.shape),
                None, np.max(indices) + 1), axis=0)]
        cluster_to_class[i] = mode

    # map cluster id to class ids
    labels_copy = np.copy(labels_predicted)
    for i in range(num_classes):
        cluster_points = np.where(labels_copy == i)
        labels_predicted[cluster_points] = cluster_to_class[i]
    
    #print('Labels true')
    #print(labels_true)
    #print('Labels predicted')
    #print(labels_predicted)

    acc = metrics.accuracy_score(labels_true, labels_predicted)
    nmi = metrics.cluster.normalized_mutual_info_score(
            labels_true, labels_predicted)
    precision = metrics.precision_score(labels_true, labels_predicted,
                                       average='micro')
    recall = metrics.recall_score(labels_true, labels_predicted,
                                  average='micro')
    f1_score = 2*precision*recall/(precision+recall)

    print('Accuracy : %f' % acc)
    print('NMI : %f' % nmi)
    print('Precision : %f' % precision)
    print('Recall : %f' % recall)
    print('F1 score : %f' % f1_score)
    print("")
    results = {
                'true' : labels_true,
                'predicted' : labels_predicted,
                'accuracy' : acc,
                'precision' : precision,
                'recall' : recall,
                'f1' : f1_score,
                'nmi' : nmi
              }
    return  results


def SaveClusterResults(base_dir, prefix, results, data_set):
    # first save stats
    with open(os.path.join(base_dir, '%s_stats' % prefix), 'w') as r:
        r.write('best accuracy : %f\n' % results['accuracy'])
        r.write('best precision : %f\n' % results['precision'])
        r.write('best recall : %f\n' % results['recall'])
        r.write('best f1 : %f\n' % results['f1'])
        r.write('best nmi : %f\n' % results['nmi'])
    # now choose a random image from each class and find which points are in
    # the cluster that the image lies in
    labels_true = results['true']
    labels_pred = results['predicted']
    unique = np.unique(labels_true)
    num_classes = len(unique)
    paths = data_set.images
    birdnames = data_set.birdnames
    with open(os.path.join(base_dir, '%s_query' % prefix), 'w') as r:
        for i in range(num_classes):
            cid = unique[i]
            # images predicted as cid
            if cid in labels_pred:
                idq1 = np.random.choice(np.where(labels_pred == cid)[0], 3)
            else:
                idq1 = np.random.choice(np.where(labels_true == cid)[0], 3)
            class_pred1 = labels_pred[idq1]
            # images that are cid
            idq2 = np.random.choice(np.where(labels_true == cid)[0], 3)
            for k in range(3):
                r.write(paths[idq1[k]])
                cc = labels_true[idq1[k]]
                cp = labels_pred[idq1[k]]
                r.write(':'+birdnames[cc]+' -> '+birdnames[cp])
                r.write(', ')
            for k in range(3):
                r.write(paths[idq2[k]])
                cc = labels_true[idq2[k]]
                cp = labels_pred[idq2[k]]
                r.write(':'+birdnames[cc]+' -> '+birdnames[cp])
                r.write(', ')
            r.write('\n')
    

def TestTriplets(test_loader, tnet, criterion):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3, _, _, _) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, distc, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        loss_triplet =  criterion(dista, distb, distc, target, args.margin, args.in_triplet_hard).data[0]
        
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        test_loss = loss_triplet + args.reg * loss_embedd

        # measure accuracy and record loss
        acc = LossAccuracy(dista, distb, distc, args.margin)
        accs.update(acc, data1.size(0))
        losses.update(test_loss.data[0], data1.size(0))      

    print('\nTest/val triplets: Average loss: %f, Accuracy: %f \n' %
            (losses.avg, accs.avg))
    return losses.avg, accs.avg

def SaveCheckpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join(runs_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(directory, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def LossAccuracy(dista, distb, distc, margin):
    margin = 0
    if args.in_triplet_hard:
        dist_neg = torch.cat([distb, distc], dim=1)
        dist_neg = torch.min(dist_neg, dim=1)[0]
    else:
        dist_neg = distb
    pred = (dista - dist_neg - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()  
