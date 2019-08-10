import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos_pred, neg_pred):
    # return 1 - (torch.sigmoid(pos_pred - neg_pred)).mean()
    return -torch.log(torch.sigmoid(pos_pred - neg_pred)).mean()


def weighted_margin_rank_batch():
    pass


def top1_loss():
    # a kind of ranking loss defined in GRU4Rec,
    # similar to bpr_loss.
    pass


def mse_loss(pred, target):
    return ((pred - target.to(torch.float32)) ** 2).mean()
