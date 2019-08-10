import torch
import torch.nn as nn


class PoincareFM(nn.Module):

    def __init__(self, n_features, n_fields, embedding_dim=10, matching_layer='linear'):
        super(PoincareFM, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_fields = n_fields
        self.n_interactions = n_fields * (n_fields - 1) // 2
        self.matching_layer = matching_layer

        self.feature_embedding = nn.Embedding(self.n_features, self.embedding_dim)
        self.feature_coeff = nn.Embedding(self.n_features, 1)
        self.bias = nn.Parameter(torch.Tensor(1))
        if self.matching_layer == 'linear':
            self.beta = nn.Parameter(torch.tensor([0.0]))
            self.c = nn.Parameter(torch.tensor([0.0]))
        # self.dropout1 = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.feature_embedding.weight.data.uniform_(-0.001, 0.001)
        self.feature_coeff.weight.data.normal_(std=0.001)
        self.bias.data.zero_()

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
        interactions = torch.zeros((batch_size, self.n_interactions), device=embedding.device)  # of shape (batch_size, m(m-1)/2)
        count = 0
        for i in range(self.n_fields):  # O(m(m-1)/2), low complexity when features are highly sparse
            for j in range(i + 1, self.n_fields):
                dist = hyperbolic_distance(embedding[:, i, :], embedding[:, j, :])
                interactions[:, count] = dist.view(-1)
                count += 1

        if self.matching_layer == 'negative':
            interactions = -interactions
        elif self.matching_layer == 'identity':
            interactions = interactions
        elif self.matching_layer == 'linear':
            interactions = self.beta * interactions + self.c

        interaction_part = interactions.sum(1)
        linear_part = coeff.sum(1).view(-1)
        output = interaction_part + linear_part + self.bias
        return output

    def riemannian_grads(self):
        self._convert_gradient(self.feature_embedding.weight)

    def _convert_gradient(self, variable):
        """ Converts hyperbolic gradient to Euclidean gradient

        @param variable (torch.Parameter): hyperbolic embedding of size (n, embedding_dim)
        """
        sqnorm = squared_norm(variable.data, dim=-1, keepdim=True)
        variable.grad.data.copy_(variable.grad.data * ((1 - sqnorm) ** 2 / 4))

    def project_embedding(self):
        renormed = torch.renorm(self.feature_embedding.weight, 2, -1, 1.0 - 1e-6)
        self.feature_embedding.weight.data.copy_(renormed)
        # self._scale_embedding(self.feature_embedding.weight)

    # def _scale_embedding(self, embedding):
    #     """ Rescale embeddings whose norm >= 1.0 to be within the Poincare dist (unit ball)

    #     @param embedding (torch.Parameter): hyperbolic embedding of shape (n, embedding_dim)
    #     """
    #     embedding_norm = torch.norm(embedding.data, dim=-1, keepdim=True)
    #     embedding_norm = torch.where(embedding_norm >= 1.0,
    #                                  embedding_norm + 1e-5,
    #                                  torch.tensor(1.0, device=embedding_norm.device))
    #     embedding.data.copy_(embedding.data / embedding_norm)

    def l2_sqnorm(self):
        return (self.feature_embedding.weight ** 2).sum()


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def squared_norm(x, dim=-1, keepdim=True):
    return (x ** 2).sum(dim=dim, keepdim=keepdim)


def hyperbolic_distance(x, y, eps=1e-7):
    """ Poincare distance function.
    """
    # z = squared_norm(x - y)  # (N, 1)
    # x_d = 1 - squared_norm(x)
    # y_d = 1 - squared_norm(y)
    # z = 2 * z / (x_d * y_d + eps) + 1
    # return arcosh(z)
    return Distance.apply(x, y).view(-1, 1)


class Distance(torch.autograd.Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps=1e-5):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps=1e-5):
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None
