import torch
import torch.nn as nn

def SimpleHingeLoss(dista, distb, distc, target, margin, hard_triplet=False):
    if hard_triplet:
        dist_neg = torch.cat([distb, distc], dim=1)
        dist_neg = torch.min(dist_neg, dim=1)[0]
    else:
        dist_neg = distb
    return nn.MarginRankingLoss(margin = margin)(dista, dist_neg, target)

def SimpleSquareHingeLoss(dista, distb, distc, target, margin, hard_triplet=False):
    if hard_triplet:
        dist_neg = torch.cat([distb, distc], dim=1)
        dist_neg = torch.min(dist_neg, dim=1)[0]
    else:
        dist_neg = distb
    return nn.MarginRankingLoss(margin = margin)(torch.pow(dista, 2), torch.pow(dist_neg, 2), target)

def RatioLoss(dista, distb, distc, target, margin, hard_triplet=False):
    if hard_triplet:
        dist_neg = torch.cat([distb, distc], dim=1)
        dist_neg = torch.min(dist_neg, dim=1)[0]
    else:
        dist_neg = distb
    ep = torch.exp(dista)
    en = torch.exp(dist_neg)
    t1 = ep/(ep+en)
    t2 = en/(ep+en)
    loss = torch.mean(torch.pow(t1, 2) + 1 - torch.pow(t2, 2))
    return loss
