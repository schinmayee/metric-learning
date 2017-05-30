import torch
import torch.nn as nn

def SimpleHingeLoss(dista, distb, distc, target, margin):
    return nn.MarginRankingLoss(margin = margin)(dista, distb, target)

def SimpleHingeLossHardTriplet(dista, distb, distc, target, margin):
    dist_neg = torch.cat([distb, distc], dim=1)
    return nn.MarginRankingLoss(margin = margin)(dista, torch.min(dist_neg, dim=1)[0], target)
