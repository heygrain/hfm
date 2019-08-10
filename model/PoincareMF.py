import torch
import torch.nn as nn
from .PoincareFM import hyperbolic_distance, squared_norm


class HyperbolicEmbedding(nn.Module):

    def __init__(self, num_users, num_items, embedding_dim=32, matching_layer='linear'):
        super(HyperbolicEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.embedding_dim)
        self.matching_layer = matching_layer
        if self.matching_layer == 'linear':
            self.beta = nn.Parameter(torch.tensor([0.0]))
            self.c = nn.Parameter(torch.tensor([0.0]))
        self.reset_parameters()

    def reset_parameters(self):
        self.user_embedding.weight.data.uniform_(-0.001, 0.001)
        self.item_embedding.weight.data.uniform_(-0.001, 0.001)

    def forward(self, user_ids, item_ids):
        u = self.user_embedding(user_ids.view(-1))
        v = self.item_embedding(item_ids.view(-1))
        d_poincare = hyperbolic_distance(u, v)
        if self.matching_layer == 'negative':
            score = -d_poincare
        elif self.matching_layer == 'identity':
            score = d_poincare
        elif self.matching_layer == 'linear':
            score = self.beta * d_poincare + self.c
        return score.view(-1)

    def riemannian_grads(self):
        self._convert_gradient(self.user_embedding.weight)
        self._convert_gradient(self.item_embedding.weight)

    def _convert_gradient(self, variable):
        """ Converts hyperbolic gradient to Euclidean gradient

        @param variable: torch.Parameter of hyperbolic embedding of size (n, embedding_dim)
        """
        sqnorm = squared_norm(variable.data, dim=-1, keepdim=True)
        variable.grad.data.copy_(variable.grad.data * ((1 - sqnorm) ** 2 / 4))

    def project_embedding(self):
        renormed_user = torch.renorm(self.user_embedding.weight, 2, -1, 1.0 - 1e-6)
        renormed_item = torch.renorm(self.item_embedding.weight, 2, -1, 1.0 - 1e-6)
        self.user_embedding.weight.data.copy_(renormed_user)
        self.item_embedding.weight.data.copy_(renormed_item)

    def l2_sqnorm(self):
        return (self.user_embedding.weight ** 2).sum() + (self.user_embedding.weight ** 2).sum()
