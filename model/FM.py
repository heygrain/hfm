import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):

    def __init__(self, n_features, embedding_dim=10):
        super(FactorizationMachine, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.feature_embedding = nn.Embedding(self.n_features, self.embedding_dim)
        self.feature_coeff = nn.Embedding(n_features, 1)
        # self.bn = nn.BatchNorm1d(self.embedding_dim)
        # self.dropout = nn.Dropout(0.5)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.feature_embedding.weight.data, std=0.01)
        nn.init.normal_(self.feature_coeff.weight.data, std=0.01)
        self.bias.data.zero_()

    def forward(self, features):
        """forward function of the FM model

        @param features: batch of input features of shape (batch_size, n_feat_ids),
                         where n_feat_ids can be different? (for now must be the same)
        """
        embedding = self.feature_embedding(features)  # of shape (batch_size, n_feat_ids, embedding_dim)
        coeff = self.feature_coeff(features)  # of shape (batch_size, n_feat_ids, 1)
        square_sum_embedding = embedding.sum(1) ** 2  # of shape (batch_size, embedding_dim)
        sum_square_embedding = (embedding ** 2).sum(1)
        interaction_part = 0.5 * (square_sum_embedding - sum_square_embedding).sum(1)
        # interaction_part = 0.5 * (square_sum_embedding - sum_square_embedding)
        # interaction_part = self.dropout(self.bn(interaction_part)).sum(1)
        linear_part = coeff.sum(1).view(-1)
        output = interaction_part + linear_part + self.bias
        return output

    def l2_sqnorm(self):
        return (self.feature_embedding.weight ** 2).sum()
