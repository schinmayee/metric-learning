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

# networks
import simplenet

# triplet loss
from tripletnet import Tripletnet

# sampling
import hard_mining

# data loader
from triplet_cub_loader import CUBTriplets

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='adam beta1 (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='adam beta2 (default: 0.999)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', type=str, default='',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', type=str, default='TripletNet',
        help='name of experiment (default: TripletNet)')
parser.add_argument('--data', type=str, default='cub-2011',
        help='dataset (default: cub-2011)')
parser.add_argument('--triplet_freq', type=int, default=5, metavar='N',
                    help='epochs before new triplets list (default: 10)')
parser.add_argument('--network', type=str, default='Simple',
        help='network architecture to use (default: Simple)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

# globals
best_acc = 0

# parameters
im_size = 64
train_classes=range(10)
test_classes=range(10,15)

triplets_per_class=16
hard_frac = 0.5

OurSampler = hard_mining.NHardestTripletSampler

# main
def main():
    global args, best_acc

    args = parser.parse_args()

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args.data == 'cub-2011':
        TLoader = CUBTriplets
        data_path = os.path.join(dir_path, 'datasets/cub-2011')

    train_data_set = TLoader(data_path,
                             n_triplets=triplets_per_class*len(train_classes),
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                             ]),
                             classes=train_classes, im_size=im_size)
    train_loader = torch.utils.data.DataLoader(
        train_data_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_data_set = TLoader(data_path,
                            n_triplets=triplets_per_class*len(test_classes),
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                            ]),
                            classes=test_classes, im_size=im_size)
    test_loader = torch.utils.data.DataLoader(
        test_data_set, batch_size=args.batch_size, shuffle=True, **kwargs)

    # network
    Net = None
    if args.network == 'Simple':
        print('Using simple net')
        Net = simplenet.Simplenet
    model = Net(im_size)

    # triplet loss
    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()

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
                         args.batch_size/8)
    
    for epoch in range(1, args.epochs + 1):
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch, sampler)
        # evaluate on validation set
        acc = test(test_loader, tnet, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

        # reset sampler and regenerate triplets every few epochs
        if epoch % args.triplet_freq == 0:
            # TODO: regenerate triplets
            train_data_set.regenerate_triplet_list(sampler, hard_frac)
            # then reset sampler
            sampler.Reset()

def train(train_loader, tnet, criterion, optimizer, epoch, sampler):
    losses = AverageMeter()
    accs = AverageMeter()
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
        
        loss_triplet = criterion(dista, distb, target)
        sampler.SampleNegatives(dista, distb, loss_triplet, (idx1, idx2, idx3))
        
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

def test(test_loader, tnet, criterion, epoch):
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
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()  
