import torch
import torch.nn as nn


class SimpleMatrixFactorization(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32):
        super(SimpleMatrixFactorization, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight.data, std=0.01)
        nn.init.normal_(self.item_embedding.weight.data, std=0.01)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids)
        v = self.item_embedding(item_ids)
        return (u * v).sum(1)

    def l2_sqnorm(self):
        return (self.user_embedding ** 2).sum() + (self.item_embedding ** 2).sum()
