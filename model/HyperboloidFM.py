import torch
from torch import nn, optim
from torch.autograd import Function
import math


class HyperFM(nn.Module):

    def __init__(self, n_features, n_fields, embedding_dim=10, matching_layer='linear'):
        super(HyperFM, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_fields = n_fields
        self.n_interactions = n_fields * (n_fields - 1) // 2
        self.matching_layer = matching_layer

        self.feature_embedding = nn.Embedding(self.n_features, 1 + self.embedding_dim)  # 1 + embedding_dim
        self.feature_coeff = nn.Embedding(self.n_features, 1)
        self.bias = nn.Parameter(torch.Tensor(1))
        if self.matching_layer == 'linear':
            self.beta = nn.Parameter(torch.tensor([0.0]))
            self.c = nn.Parameter(torch.tensor([0.0]))
        # self.dropout1 = nn.Dropout(0.5)
        self.reset_parameters()

    def reset_parameters(self):
        self.feature_embedding.weight.data.uniform_(-0.001, 0.001)
        self.project_embedding()
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
        """ Converts hyperbolic gradient to Euclidean gradient (hm & Pi_p(d_p))

        @param variable (torch.Parameter): hyperbolic embedding of size (n, embedding_dim)
        """
        u = self.feature_embedding.weight.grad
        x = self.feature_embedding.weight.data
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(ldot(x, u, keepdim=True).expand_as(x), x)
        return u  # can be delete?

    def project_embedding(self):
        """Normalize vector such that it is located on the hyperboloid"""
        w = self.feature_embedding.weight.data
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        tmp = 1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        tmp.sqrt_()
        w.narrow(-1, 0, 1).copy_(tmp)
        return w  # can be delete?

    def l2_sqnorm(self):
        return (self.feature_embedding.weight ** 2).sum()

    def to_poincare_ball(self):
        x = self.feature_embedding.weight.data.clone()
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)


def hyperbolic_distance(u, v, keepdim=False):
    """Lorentz distance"""
    d = -ldot(u, v)
    d.data.clamp_(min=1)  # clamp elements into range [min, infty)
    return acosh(d, 1e-5)


def ldot(u, v, keepdim=False):
    """Lorentzian scalar product"""
    uv = u * v
    uv.narrow(-1, 0, 1).mul_(-1)
    return torch.sum(uv, dim=-1, keepdim=keepdim)


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply


class RSGD(optim.Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(RSGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                step_size = -group["lr"] * p.grad
                if group["hyperboloid"]:
                    self.expm(p.data, step_size)
                else:
                    p.data.add_(step_size)

    def expm(self, p, d_p):
        """Exponential map for hyperboloid"""
        ldv = ldot(d_p, d_p, keepdim=True)
        nd_p = ldv.clamp_(min=0).sqrt_()
        t = torch.clamp(nd_p, max=1)  # ? max clip
        nd_p.clamp_(min=1e-5)
        newp = (torch.cosh(t) * p).addcdiv_(torch.sinh(t) * d_p, nd_p)
        p.copy_(newp)


class RAdam(optim.Adam):
    def __init__(self, params, lr):
        super(RAdam, self).__init__(params, lr)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = -group['lr'] * math.sqrt(bias_correction2) / bias_correction1 * exp_avg / denom

                if group["hyperboloid"]:
                    self.expm(p.data, step_size)
                else:
                    p.data.add_(step_size)

        return loss

    def expm(self, p, d_p):
        """Exponential map for hyperboloid"""
        ldv = ldot(d_p, d_p, keepdim=True)
        # if False:
        #     assert all(ldv > 0), "Tangent norm must be greater 0"
        #     assert all(ldv == ldv), "Tangent norm includes NaNs"
        nd_p = ldv.clamp_(min=0).sqrt_()
        t = torch.clamp(nd_p, max=1)  # ? max clip
        nd_p.clamp_(min=1e-5)
        newp = (torch.cosh(t) * p).addcdiv_(torch.sinh(t) * d_p, nd_p)
        p.copy_(newp)
