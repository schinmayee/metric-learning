import torch
import torch.nn as nn
import torch.nn.functional as F

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

def EmbHingeLoss(emba, embb, embc, margin, target):
    triplet_loss = nn.functional.triplet_margin_loss(
	    emba, embb, embc, margin=margin)
    return triplet_loss

def EmbSquareHingeLoss(emba, embb, embc, margin, target):
    dist_pos = F.pairwise_distance(emba, embb, 2)
    dist_neg = F.pairwise_distance(emba, embc, 2)
    triplet_loss = nn.MarginRankingLoss(margin = margin)(torch.pow(dist_pos, 2), torch.pow(dist_neg, 2), target)
    return triplet_loss

def EmbSoftHingeLoss(emba, embb, embc, margin, target):
    dist_pos  = F.pairwise_distance(emba, embb, 2)
    dist_neg1 = F.pairwise_distance(emba, embc, 2)
    dist_neg2 = F.pairwise_distance(embb, embc, 2)
    dist_neg_s = -torch.log(torch.exp(margin - dist_neg1) + torch.exp(margin - dist_neg2))
    loss = nn.MarginRankingLoss(margin = 0)(dist_pos, dist_neg_s, target)
    return loss
