import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import numpy as np
from random import shuffle

# networks
import model_net

# triplet loss
import triplet_net

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

parser.add_argument('--name', type=str, default='TripletNet',
        help='name of experiment (default: TripletNet)')
parser.add_argument('--data', type=str, default='cub-2011',
        help='dataset (default: cub-2011)')

parser.add_argument('--triplet_freq', type=int, default=10,
                    help='epochs before new triplets list (default: 10)')
parser.add_argument('--val-freq', type=int, default=2,
        help='epochs before validating on validation set (default: 2)')

parser.add_argument('--network', type=str, default='Simple',
        help='network architecture to use (default: Simple)')
parser.add_argument('--log-interval', type=int, default=2,
        help='how many batches to wait before logging training status (default: 2)')
parser.add_argument('--feature-size', type=int, default=64,
        help='size for embeddings/features to learn')

# globals
best_acc = 0
feature_size = 0

# parameters
im_size = 64
num_train=4
num_val=4
num_test=1
train_classes=range(num_train)  # triplets_per_class*train_classes should be a multiple of batch size (64 by default)
val_classes=range(num_train,num_train+num_val)
test_classes=range(num_train+num_val,num_train+num_val+num_test)

triplets_per_class=16  # keep at least 16 triplets per class, later increase to 32/64
hard_frac = 0.5

OurSampler = hard_mining.NHardestTripletSampler

runs_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        ('runs/r-%s' % time.strftime('%m-%d-%H-%M')))

# main
def main():
    global args, best_acc, feature_size, im_size

    args = parser.parse_args()

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
    model = Net(feature_size=feature_size, im_size=im_size)

    # triplet loss
    tnet = triplet_net.TripletNet(model)
    if args.cuda:
        tnet.cuda()

    # data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.data == 'cub-2011':
        TLoader = CUBTriplets
        DLoader = CUBImages
        data_path = os.path.join(dir_path, 'datasets/cub-2011')

    train_data_set = TLoader(data_path,
                             n_triplets=triplets_per_class*len(train_classes),
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                             ]),
                             classes=train_classes, im_size=im_size)
    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True, **kwargs)

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

    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    optimizer = optim.Adam(tnet.parameters(), lr=args.lr,
                           betas=[args.beta1,args.beta2])

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    sampler = OurSampler(len(train_classes),
                         int((hard_frac+hard_frac/2)*args.batch_size))
    
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        Train(train_loader, tnet, criterion, optimizer, epoch, sampler)

        # evaluate on validation set
        if epoch % args.val_freq == 0:
            acc = TestTriplets(val_loader_t, tnet, criterion)

            # remember best acc and save checkpoint
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            SaveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': tnet.state_dict(),
                'best_prec1': best_acc,
            }, is_best)

            ComputeClusters(val_loader, model, len(val_classes))

        # reset sampler and regenerate triplets every few epochs
        if epoch % args.triplet_freq == 0:
            # TODO: regenerate triplets
            train_data_set.regenerate_triplet_list(sampler, hard_frac)
            # then reset sampler
            sampler.Reset()

def Train(train_loader, tnet, criterion, optimizer, epoch, sampler):
    losses = AverageMeter()
    loss_accs = AverageMeter()
    emb_norms = AverageMeter()
    
    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3, idx1, idx2, idx3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        # forward pass
        loss_triplet = criterion(dista, distb, target)
        # sample hard ngatives based on loss
        sampler.SampleNegatives(dista, distb, loss_triplet, (idx1, idx2, idx3))
        
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + args.reg * loss_embedd

        # measure loss accuracy and record loss
        loss_acc = LossAccuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        loss_accs.update(loss_acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: compute embeddings
        # TODO: compute k-means cluster on training set
        # TODO: determine hard negatives -- points in the wrong cluster

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Loss Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
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
            data1 = data.cuda()
        data = Variable(data)

        # compute embeddings
        f = enet(data)
        embeddings[ids.numpy(),:] = f.data.numpy()
        labels_true[ids.numpy()] = classes.numpy()
    print('Generated embeddings, now running k-means for %d clusters...' % num_clusters)

    # map classes to (0,N)
    # also get initial centroids
    labels_copy = np.copy(labels_true)
    unique_classes = np.unique(labels_copy)
    classes_mapped = np.arange(0,len(unique_classes),1)
    initial_centers = np.zeros(shape=(num_clusters, feature_size), dtype=float)
    for i in range(len(unique_classes)):
        c_ids = np.where(labels_copy == unique_classes[i])
        labels_true[c_ids] = classes_mapped[i]
        use_im = np.random.choice(c_ids[0])
        #print(embeddings[use_im,:])
        initial_centers[i,:] = embeddings[use_im,:]

    kmeans_model = KMeans(n_clusters=num_clusters, random_state=1,
                          max_iter=1000, tol=1e-3,
                          init=initial_centers)
    labels_predicted = kmeans_model.fit_predict(embeddings)
    
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
                'f1' : f1_score
              }
    return 
    

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
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = LossAccuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest/val triplets: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg

def SaveCheckpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = os.path.join(runs_dir, args.name)
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

def LossAccuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()  
