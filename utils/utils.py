import torch
import pickle
import random
import numpy as np


def load_data(name, device, use_content, use_rating, path='./utils/datasets/', print_info=False):
    with open(path + name + '.processed.pkl', 'rb') as f:
        data = pickle.load(f)

    if print_info:
        print(name, 'dataset:')

    if not use_rating:
        data['train_rating'] = None
        data['dev_rating'] = None
        data['test_rating'] = None

    if not use_content:
        data['context'] = None
        data['user_demo'] = None
        data['item_demo'] = None

    n_features = data['item_idx_max']

    if data['user_demo'] is not None:
        user_demo_dim = data['user_demo'].shape[1]
        if n_features < data['user_demo'].max():
            n_features = data['user_demo'].max()
    else:
        user_demo_dim = 0

    if data['item_demo'] is not None:
        item_demo_dim = data['item_demo'].shape[1]
        if n_features < data['item_demo'].max():
            n_features = data['item_demo'].max()
    else:
        item_demo_dim = 0

    n_fields = 2 + user_demo_dim + item_demo_dim
    n_features += 1

    for key, val in data.items():
        if (val is not None) and \
                (key != 'item_idx_min') and \
                (key != 'item_idx_max'):
            data[key] = torch.from_numpy(data[key]).to(device=device)
            if print_info:
                print('\t', key, ':', val.shape)

    return data, n_fields, n_features


def batcher(X_, y_=None, batch_size=-1, shuffle=False):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError("Parameter batch_size={} is unsupported".format(batch_size))

    if shuffle:
        perm = torch.randperm(X_.shape[0])
        X_ = X_[perm]
        if y_ is not None:
            y_ = y_[perm]

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        if y_ is None:
            yield ret_x
        else:
            ret_y = y_[i:upper_bound]
            yield (ret_x, ret_y)


def construct_features(users, items, item_idx_min, context=None, user_demo=None, item_demo=None):
    features = torch.cat([users, items], dim=1)
    if context is not None:
        features = torch.cat([features, context], dim=1)  # context is sampled together with users
    if user_demo is not None:
        ufeat = user_demo[users.reshape(-1)]
        features = torch.cat([features, ufeat], dim=1)
    if item_demo is not None:
        ifeat = item_demo[items.reshape(-1) - item_idx_min]
        features = torch.cat([features, ifeat], dim=1)
    return features


def set_random_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def save_results(file, args, dev_score, test_score, dev_hit5, dev_hit10, test_hit5, test_hit10):
    score_string = 'dAUC: {:.5f}'.format(dev_score) + \
                   '\ttAUC: {:.5f}'.format(test_score) + \
                   '\tdHR5: {:.5f}'.format(dev_hit5) + \
                   '\ttHR5: {:.5f}'.format(test_hit5) + \
                   '\tdHR10: {:.5f}'.format(dev_hit10) + \
                   '\ttHR10: {:.5f}'.format(test_hit10) + \
                   '\tdim: ' + str(args.embedding_dim) + \
                   '\tlr: ' + str(args.lr) + \
                   '\tlr_R: ' + str(args.lr_poincare) + \
                   '\treg: ' + str(args.reg_l2) + \
                   '\tdata: ' + str(args.dataset) + \
                   '\n'
    with open(file, 'a') as f:
        f.write(score_string)


def recall_at(prediction, target):
    """ Compute average recall@k, also called HR@k (hit ratio)

    @param prediction: (n_test_users, n_items)
    @param target: groud truth of users' clicks (n_test_users, )
    @param k: k
    """
    rank_target = (prediction.t() > prediction[:, target]).sum(0) + 1.0
    recall5 = (rank_target <= 5).to(torch.float32).mean()
    recall10 = (rank_target <= 10).to(torch.float32).mean()
    return recall5, recall10


def hit_eval(model, data, on='test'):
    print('Evaluate on', on, 'set',)
    key_user = on + '_user'
    key_item = on + '_item'
    user, pos_item = data[key_user], data[key_item]    
    model.eval()
    batch_size = 1024
    with torch.no_grad():
        epoch_score5 = torch.tensor(0.0, device=user.device)
        epoch_score10 = torch.tensor(0.0, device=user.device)
        for i, (u, pos) in enumerate(batcher(user, pos_item, batch_size=batch_size, shuffle=False)):
            num_user = u.shape[0]
            neg = torch.randint(low=data['item_idx_min'], high=data['item_idx_max'] + 1, size=(num_user, 99), device=user.device)
            pn = torch.cat([pos, neg], dim=1).view((-1, 1))
            u = u.repeat(1, 100).view((-1, 1))
            feat = construct_features(u, pn, data['item_idx_min'], user_demo=data['user_demo'], item_demo=data['item_demo'])   # (N*100, p)
            pred = model(feat)
            pred = pred.view((num_user, 100))
            score5, score10 = recall_at(pred, target=0)
            epoch_score5 += score5 * num_user
            epoch_score10 += score10 * num_user
        score5 = epoch_score5 / user.shape[0]
        score10 = epoch_score10 / user.shape[0]
        print("-", on, "score5: {:.5f}".format(score5), "score10: {:.5f}".format(score10))
        print()
    return score5, score10
