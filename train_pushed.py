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
import losses

from cub_loader import CUBImages
from class_sampler import ClassSampler

# sklearn for clustering, KNN and evaluating clusters
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

parser.add_argument('--loss', type=str, default='Hinge',
	help='loss mechanism (default: Hinge, other options: Square, Soft)')
parser.add_argument('--data', type=str, default='cub-2011',
        help='dataset (default: cub-2011)')

parser.add_argument('--val-freq', type=int, default=1,
        help='epochs before validating on validation set (default: 1)')
parser.add_argument('--results-freq', type=int, default=1,
        help='epochs before saving results (default: 1)')

parser.add_argument('--network', type=str, default='Simple',
        help='network architecture to use (default: Simple)')
parser.add_argument('--log-interval', type=int, default=2,
        help='how many batches to wait before logging training status (default: 2)')
parser.add_argument('--feature-size', type=int, default=64,
        help='size for embeddings/features to learn')

parser.add_argument('--num-train', type=int, default=8,
        help='Number of train classes')
parser.add_argument('--num-val', type=int, default=4,
        help='Number of validation classes')
parser.add_argument('--num-test', type=int, default=4,
        help='Number of test classes')
parser.add_argument('--num-species', type=int, default=4,
        help='Number of specie in each batch')
parser.add_argument('--num-batches', type=int, default=100,
	help='Number of batches per epoch (...)')
parser.add_argument('--num-clusters', type=int, default=1,
	help='Number of clusters per unknown class')

parser.add_argument('--normalize-features', action='store_true', default=False,
                    help='normalize features')
parser.add_argument('--local-sampling', action='store_true', default=False,
                    help='use local positive sampling')

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

hard_frac = 0.6

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
            ('runs/rp-%s-f%d-%s' %
                (args.network, args.feature_size, time.strftime('%m-%d-%H-%M'))))
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
        val_classes=range(num_train-num_val,num_train)
        test_classes=range(num_train,num_train+num_test)

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
	## force feature size to be 2048
	#feature_size = 2048
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

    if args.cuda:
        model.cuda()

    # data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.data == 'cub-2011':
        DLoader = CUBImages
        data_path = os.path.join(dir_path, 'datasets/cub-2011')
    else:
        assert(False)

    train_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=train_classes, im_size=im_size)
    train_loader = torch.utils.data.DataLoader(
            train_data_set, batch_size=args.batch_size, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(train_data_set),
            **kwargs)

    val_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=val_classes, im_size=im_size)
    val_loader = torch.utils.data.DataLoader(
            val_data_set, batch_size=args.batch_size/2, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(val_data_set),
            **kwargs)

    test_data_set = DLoader(data_path,
                           transform=transforms.Compose([
                             transforms.ToTensor(),
                           ]),
                           classes=test_classes, im_size=im_size)
    test_loader = torch.utils.data.DataLoader(
            test_data_set, batch_size=args.batch_size/3, shuffle=False, 
            sampler=torch.utils.data.sampler.SequentialSampler(test_data_set),
            **kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    net_params = model.SetLearningRate(args.lr*0.1, args.lr)
    optimizer = optim.Adam(net_params, lr=args.lr,
                           betas=[args.beta1,args.beta2])

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    labels_true = None
    labels_predicted = None
    assert(args.batch_size % args.num_species == 0)
    num_species = args.num_species
    num_per_specie = args.batch_size/args.num_species

    for epoch in range(1, args.epochs + 1):

	train_data_set_t = DLoader(data_path,
    	                         transform=transforms.Compose([
    	                           transforms.ToTensor(),
    	                         ]),
    	                         classes=train_classes, im_size=im_size)
    	train_loader_t = torch.utils.data.DataLoader(
	    train_data_set_t, batch_size=args.batch_size,
	    sampler=ClassSampler(
		train_data_set_t, num_species,
		num_per_specie, args.num_batches), **kwargs)

        # train for one epoch
        train_loss = Train(train_loader_t, model, optimizer, epoch,
		           num_species, num_per_specie)

        # evaluate on validation set
        if epoch % args.val_freq == 0:
	    val_embeddings, val_labels_true = ComputeEmbeddings(val_loader, model)
            val_results = ComputeClusters(val_embeddings, val_labels_true)
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
                'state_dict': model.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

            # save data for 2 plots here:
            #   1. train and test loss (triplet)
            #   2. triplet and cluster based classification accuracy on validation set
            epochs.append(epoch)
            train_losses.append(train_loss)
            classification_accs.append(acc)

	    # save final results and plot loss/accuracy with training
            print('Results are being saved to %s' % runs_dir)
            pickle.dump(epochs, open(os.path.join(runs_dir, 'epochs'), 'w'))
            pickle.dump(train_losses, open(os.path.join(runs_dir, 'train_losses'), 'w'))
            pickle.dump(classification_accs, open(os.path.join(runs_dir, 'classification_accs'), 'w'))

            # at the end, save some query results for visualization
            SaveClusterResults(runs_dir, 'val', val_results, val_data_set)

        if epoch % args.results_freq == 0:
            # also run the model and kmeans over test data and save the results
            # over test data, BUT DO NOT use this for tuning hyper-parameters
            print('Saving test results!!')
	    test_embeddings, test_labels_true = ComputeEmbeddings(test_loader, best_model)
            test_results = ComputeClusters(test_embeddings, test_labels_true)
            SaveClusterResults(runs_dir, 'test', test_results, test_data_set)
	    for num_neighbors in [1,2,4,8,16,32]:
		accuracy, recall, query_results = ComputeKNN(
			test_embeddings, test_labels_true,
		        num_neighbors=num_neighbors)
		SaveKNNResults(runs_dir, 'test', num_neighbors, accuracy,
			       recall, query_results, test_data_set)
		print('KNN for %d neighbors: accuracy = %f, recall = %f' %
			(num_neighbors, accuracy, recall))
	    print('...\n')
            

def Train(train_loader_t, model, optimizer, epoch,
	  num_species, num_per_specie):
    losses = AverageMeter()
    emb_norms = AverageMeter()
    
    # switch to train mode
    model.train()

    loss_triplet = 0
    loss_embedd = 0

    for batch_idx, (data, labels, idx) in enumerate(train_loader_t):
        if args.cuda:
            data = data.cuda()
	data = Variable(data)

        # compute output
        embed = model(data)
	loss_embed = embed.norm(2)
	loss_triplet = ComputeTripletLoss(embed, labels,
		                          num_species, num_per_specie)

        loss = loss_triplet + args.reg * loss_embed

        # measure loss accuracy and record loss
        losses.update(loss_triplet.data[0], data.size(0))
        emb_norms.update(loss_embed.data[0]/3, data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{}]\t'
	      'Loss: {:.4f} \t'
              'Total Loss: {:.2f}'.format(
            epoch, (batch_idx+1) * len(data), len(train_loader_t.sampler),
            loss_triplet.data[0], loss.data[0]))

def ComputeTripletLoss(features, labels, num_species, num_per_specie):
    features = features.cpu()
    anchor = list()
    positive = list()
    negative = list()
    start_idx = 0
    #print('Computing triplets ...')
    #total_loss = 0
    for i in range(num_species):
	threshold = feature_size*10.0 # some large number for threshold, for the default case
	if args.local_sampling:
	    anc_pos = list()
	    # go over all anchor-positive pairs and compute distances
	    for j in range(num_per_specie):
		aidx = start_idx + j
		for pair in range(j, num_per_specie):
		    pidx = start_idx + pair
		    ap_dist = (features[aidx]-features[pidx]).norm(2).data[0]
		    anc_pos.append(ap_dist)
	    anc_pos = np.array(anc_pos)
	    # sort and select some threshold distance for positive sampling
	    anc_pos.sort()
	    threshold = anc_pos[int(anc_pos.size*0.6)]

	for j in range(num_per_specie):
	    aidx = start_idx + j
	    diff = features - features[aidx,:].expand_as(features)
	    diff_norms = diff.norm(2, 1).squeeze()
	    for pair in range(j, num_per_specie):
		pidx = start_idx + pair
		ap_dist = (features[aidx]-features[pidx]).norm(2).data[0]
		if ap_dist > threshold:  # skip this triplet
		    continue
		ap_dist = torch.Tensor(feature_size).fill_(ap_dist)
		norms_loss = diff_norms.data - ap_dist  # negative - positive-dist
		norms_loss[start_idx:start_idx+num_per_specie] = 2*args.margin
		in_margin = norms_loss.lt(args.margin)
		all_neg = in_margin.nonzero()
		if (len(all_neg.size()) == 0):
		    continue
		#this_loss = torch.sum(-norms_loss[in_margin] + args.margin)
		#print('This loss %f' % this_loss)
		for k in range(all_neg.size()[0]):
		    anchor.append(features[aidx].view(1,-1))
		    positive.append(features[pidx].view(1,-1))
		    negative.append(features[all_neg[k]].view(1,-1))
		#total_loss += this_loss
	start_idx += num_per_specie
    #print('Number of triplets = %d' % len(anchor))
    #print('Total loss should be %f' % (total_loss/len(anchor)))
    av = torch.cat(anchor, dim=0).view(len(anchor), feature_size)
    pv = torch.cat(positive, dim=0).view(len(positive), feature_size)
    nv = torch.cat(negative, dim=0).view(len(negative), feature_size)
    if args.cuda:
	av = av.cuda()
	pv = pv.cuda()
	nv = nv.cuda()
    target = torch.FloatTensor(av.size()[0]).fill_(1)
    if args.cuda:
        target = target.cuda()
    target = Variable(target)
    if args.loss == 'Hinge':
	triplet_loss = losses.EmbHingeLoss(av, pv, nv, args.margin, target)
    elif args.loss == 'Square':
	triplet_loss = losses.EmbSquareHingeLoss(av, pv, nv, args.margin, target)
    elif args.loss == 'Soft':
	triplet_loss = losses.EmbSoftHingeLoss(av, pv, nv, args.margin, target)
    return triplet_loss

def ComputeEmbeddings(loader, enet):
    global feature_size
    enet.eval()
    embeddings = np.zeros(shape=(len(loader.dataset), feature_size),
                          dtype=float)
    labels_true = np.zeros(shape=(len(loader.dataset)), dtype=int)
    for batch_idx, (data, classes, ids) in enumerate(loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data)

        # compute embeddings
        f = enet(data)
        embeddings[ids.numpy(),:] = f.cpu().data.numpy()
        labels_true[ids.numpy()] = classes.cpu().numpy()
    return embeddings, labels_true

def ComputeClusters(embeddings, labels_true):
    global feature_size
    num_clusters = args.num_clusters
    print('Running k-means for %d clusters per class ...' % num_clusters)

    # initialize centroid(s)  for each cluster
    unique_classes = np.unique(labels_true)
    num_classes = len(unique_classes)
    initial_centers = np.zeros(shape=(num_classes*num_clusters, feature_size), dtype=float)
    for i in range(num_classes):
	for ii in range(num_clusters):
	    c_ids = np.where(labels_true == unique_classes[i])
	    use_im = np.random.choice(c_ids[0])
	    initial_centers[i*num_clusters+ii,:] = embeddings[use_im,:]

    kmeans_model = KMeans(n_clusters=num_classes*num_clusters, random_state=1,
                          max_iter=1000, tol=1e-3,
                          init=initial_centers, n_init=1)
    labels_predicted = kmeans_model.fit_predict(embeddings)

    # map predicted clusters to actual class ids
    cluster_to_class = np.zeros(shape=(num_classes*num_clusters,), dtype=int)
    for i in range(num_classes*num_clusters):
        # figure out which class this cluster must be
        # 1. set of points that belong to this cluster
        cluster_points = np.where(labels_predicted == i)
        # 2. true class labels for these points
        actual_labels = labels_true[cluster_points]
        # 3. map cluster to most frequently occuring class
        unique, indices = np.unique(actual_labels, return_inverse=True)
        mode = unique[np.argmax(
            np.apply_along_axis(
                np.bincount, 0, indices.reshape(actual_labels.shape),
                None, np.max(indices) + 1), axis=0)]
        cluster_to_class[i] = mode

    # map cluster id to class ids
    labels_copy = np.copy(labels_predicted)
    for i in range(num_classes*num_clusters):
        cluster_points = np.where(labels_copy == i)
        labels_predicted[cluster_points] = cluster_to_class[i]
    
    acc = metrics.accuracy_score(labels_true, labels_predicted)
    nmi_s = metrics.cluster.normalized_mutual_info_score(
              labels_true, labels_predicted)
    mi = metrics.cluster.mutual_info_score(
            labels_true, labels_predicted
            )
    h1 = metrics.cluster.entropy(labels_true)
    h2 = metrics.cluster.entropy(labels_predicted)
    nmi = 2*mi/(h1+h2)
    print(mi, h1, h2)
    precision = metrics.precision_score(labels_true, labels_predicted,
                                       average='micro')
    recall = metrics.recall_score(labels_true, labels_predicted,
                                  average='micro')
    f1_score = 2*precision*recall/(precision+recall)

    print('Accuracy : %f' % acc)
    print('NMI : %f vs old %f' % (nmi, nmi_s))
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

def ComputeKNN(embeddings, labels_true, num_neighbors = 4):
    global feature_size
    print('Running KNN on all images')
    #select_ids = np.random.choice(range(len(labels_true)), num_times)
    query_results = list()
    accuracy = 0
    recall = 0
    num_times = len(labels_true)
    for i in range(num_times):
	id_query = i
	l_query = labels_true[id_query]
	f_query = embeddings[id_query, :]
	d_query = embeddings - f_query
	d_query = np.linalg.norm(d_query, axis=1).reshape((-1))
	sort_ids = np.argsort(d_query)
	res_ids = sort_ids[0:num_neighbors+1]
	res_labels = labels_true[res_ids]
	num_correct = np.where(res_labels == l_query)[0].shape[0] - 1
	query_results.append((id_query, res_ids, res_labels))
	if num_correct > 0:
	    accuracy += float(num_correct)/float(num_neighbors)
	if num_correct > 0:
	    recall += 1
    accuracy /= float(num_times)
    recall = float(recall)/float(num_times)
    return accuracy, recall, query_results

def SaveKNNResults(base_dir, prefix, num_nbrs, accuracy, recall, results, data_set):
    with open(os.path.join(base_dir, '%s_knn_%d_stats' % (prefix, num_nbrs)), 'w') as r:
	r.write('knn accuracy : %f\n' % accuracy)
	r.write('knn recall : %f\n' % recall)
    paths = data_set.images
    birdnames = data_set.birdnames
    labels = data_set.labels
    with  open(os.path.join(base_dir, '%s_knn_%d_query' % (prefix, num_nbrs)), 'w') as r:
	for q in results:
	    q_path = paths[q[0]]
	    q_name = birdnames[labels[q[0]]]
	    r.write(q_path+':'+q_name)
	    for qq in q[1]:
		qq_path = paths[qq]
		qq_name = birdnames[labels[qq]]
		r.write(','+qq_path+':'+qq_name)
	    r.write('\n')

    
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
