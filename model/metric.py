import torch
from .loss import mse_loss


def mrr_at(prediction, target, k=10):
    """ Compute mean reciprocal rank

    @param prediction: (n_test_users, n_items)
    @param target: groud truth of users' clicks (n_test_users, )
    """
    rank_target = (prediction.t() > prediction[range(prediction.shape[0]), target]).sum(0) + 1.0
    rr = 1.0 / rank_target.to(torch.float32)
    rr[rank_target > k] = 0.0
    mrr = rr.mean()
    return mrr


def recall_at(prediction, target, k=10):
    """ Compute average recall@k, also called HR@k (hit ratio)

    @param prediction: (n_test_users, n_items)
    @param target: groud truth of users' clicks (n_test_users, )
    @param k: k
    """
    rank_target = (prediction.t() > prediction[range(prediction.shape[0]), target]).sum(0) + 1.0
    recall = (rank_target <= k).to(torch.float32).mean()
    return recall


def nDCG_at(prediction, target, k=10):
    """ Compute average nDCG@k (normalized discounted cumulative gain)

    @param prediction: (n_test_users, n_items)
    @param target: groud truth of users' clicks (n_test_users, )
    @param k: k
    """
    pass


def AUC(pos_pred, neg_pred):
    return (pos_pred > neg_pred).to(torch.float32).mean()


# def AUC(prediction, target):
#     return (prediction.t() < prediction[range(prediction.shape[0]), target]).sum(0).to(torch.float32).mean()


def mse(y1, y2):
    return mse_loss(y1, y2)


def rmse(y1, y2):
    return torch.sqrt(mse_loss(y1, y2))
