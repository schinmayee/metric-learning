import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_ap = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_an = F.pairwise_distance(embedded_x, embedded_z, 2)
        dist_pn = F.pairwise_distance(embedded_y, embedded_z, 2)
        return dist_ap, dist_an, dist_pn, embedded_x, embedded_y, embedded_z

    def SetLearningRate(self, lr1, lr2):
        return self.embeddingnet.SetLearningRate(lr1, lr2)
