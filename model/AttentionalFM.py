import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionalFM(nn.Module):

    def __init__(self, n_features, n_fields, embedding_dim=10):
        super(AttentionalFM, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_fields = n_fields
        self.n_interactions = n_fields * (n_fields - 1) // 2

        self.feature_embedding = nn.Embedding(self.n_features, self.embedding_dim)
        self.feature_coeff = nn.Embedding(self.n_features, 1)
        self.bias = nn.Parameter(torch.Tensor(1))

        # attention network (a multi-layer perceptron)
        self.attention = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim // 2, bias=True)
        self.dropout1 = nn.Dropout(0.0)
        self.p = nn.Parameter(torch.Tensor(self.embedding_dim // 2, 1))
        self.w = nn.Parameter(torch.Tensor(self.embedding_dim, 1))

        self.dropout2 = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.feature_embedding.weight.data.normal_(std=0.01)
        self.feature_coeff.weight.data.normal_(std=0.01)
        self.bias.data.zero_()

        torch.nn.init.kaiming_normal_(self.attention.weight.data, nonlinearity='relu')
        self.attention.bias.data.zero_()
        self.p.data.normal_(std=1)

        torch.nn.init.constant_(self.w.data, 1)

    def forward(self, features):
        """forward function of the FM model

        @param features: batch of input features of shape (batch_size, n_feat_ids),
                         where n_feat_ids can be different? (for now must be the same)
        """
        # embedding layer
        embedding = self.feature_embedding(features)  # of shape (batch_size, n_feat_ids, embedding_dim)
        coeff = self.feature_coeff(features)  # of shape (batch_size, n_feat_ids, 1)

        # interaction layer
        batch_size = embedding.size(0)
        interactions = torch.zeros((batch_size, self.n_interactions, self.embedding_dim), device=embedding.device)  # of shape (batch_size, m(m-1)/2, embedding_dim)
        count = 0
        for i in range(self.n_fields):  # O(m(m-1)/2), low complexity when features are highly sparse
            for j in range(i + 1, self.n_fields):
                product = embedding[:, i, :] * embedding[:, j, :]
                interactions[:, count, :] = product
                count += 1
        # interactions_values = interactions.sum(2)  # of shape (batch_size, m(m-1)/2)

        # attention network
        attention_coeff = F.softmax(torch.mm(F.relu(self.attention(interactions.view(-1, self.embedding_dim))),
                                             self.p).view(batch_size, self.n_interactions, 1),
                                    dim=1)  # of shape (batch_size, m(m-1)/2, 1)
        attention_coeff = self.dropout1(attention_coeff)
        interaction_part = (interactions * attention_coeff).sum(1)  # of shape (batch_size, embedding_dim)
        interaction_part = self.dropout2(interaction_part)

        interaction_part = interaction_part.mm(self.w).view(-1)

        linear_part = coeff.sum(1).view(-1)
        output = interaction_part + linear_part + self.bias
        return output

    def l2_sqnorm(self):
        return (self.feature_embedding.weight ** 2).sum() + (self.attention.weight ** 2).sum()
