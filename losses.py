import torch
import torch.nn as nn

def SimpleHingeLoss(dista, distb, distc, target, margin, hard_triplet=False):
    if hard_triplet:
        dist_neg = torch.cat([distb, distc], dim=1)
        return nn.MarginRankingLoss(margin = margin)(dista, torch.min(dist_neg, dim=1)[0], target)
    else:
        return nn.MarginRankingLoss(margin = margin)(dista, distb, target)

def SimpleSquareHingeLoss(dista, distb, distc, target, margin, hard_triplet=False):
    if hard_triplet:
        dist_neg = torch.cat([distb, distc], dim=1)
        dist_neg = torch.min(dist_neg, dim=1)[0]
        return nn.MarginRankingLoss(margin = margin)(torch.pow(dista, 2), torch.pow(dist_neg, 2), target)
    else:
        return nn.MarginRankingLoss(margin = margin)(torch.pow(dista, 2), torch.pow(distb, 2), target)
