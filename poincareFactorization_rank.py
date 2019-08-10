import torch
from torch import optim
import argparse
import os
import time
from datetime import datetime

from model import HyperbolicEmbedding
from model.loss import bpr_loss
from model.metric import AUC
from utils.utils import load_data, set_random_seed, save_results, recall_at


def batcher(u_, p_, n_, batch_size=-1, shuffle=False):
    n_samples = u_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError("Parameter batch_size={} is unsupported".format(batch_size))

    if shuffle:
        perm = torch.randperm(u_.shape[0])
        u_ = u_[perm]
        p_ = p_[perm]
        n_ = n_[perm]

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_u = u_[i:upper_bound]
        ret_p = p_[i:upper_bound]
        ret_n = n_[i:upper_bound]
        yield (ret_u, ret_p, ret_n)


def train(model, data, output_path, args, loss_func, score_func):

    # optimizer
    all_parameters = model.parameters()
    poincare_parameters = []
    for pname, p in model.named_parameters():
        if 'embedding' in pname:
            poincare_parameters.append(p)
    poincare_parameters_id = list(map(id, poincare_parameters))
    other_parameters = list(filter(lambda p: id(p) not in poincare_parameters_id,
                            all_parameters))
    params = [{'params': poincare_parameters, 'lr': args.lr_poincare}, {'params': other_parameters, 'lr': args.lr}]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'Momentum':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.8)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(params, lr=args.lr)

    # train
    print('before train:')
    best_dev_score = evaluate(model, data, score_func, on='dev')
    best_epoch = -1
    print()

    for epoch in range(args.n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, args.n_epochs))
        train_for_epoch(model, data, loss_func, optimizer, args.batch_size, args.reg_l2)

        # evaluate on dev_set
        cond1 = not(epoch % args.eval_every) and args.eval_every > 0
        cond2 = epoch == args.n_epochs and not args.eval_every
        if cond1 or cond2:
            dev_score = evaluate(model, data, score_func, on='dev')
            if best_dev_score < dev_score:
                print("New best dev score! Saving model.")
                torch.save(model.state_dict(), output_path)
                best_dev_score = dev_score
                best_epoch = epoch
            if epoch >= best_epoch + 5:
                print("Early stopping at epoch {:}.".format(epoch + 1))
                print()
                break
        print()

    print("- Best epoch: {:}, best dev score: {:.5f}.".format(best_epoch + 1, best_dev_score))
    return best_dev_score


def train_for_epoch(model, data, loss_func, optimizer, batch_size, reg_l2):
    train_user, train_item = data['train_user'], data['train_item']
    neg_item = torch.randint(low=data['item_idx_min'], high=data['item_idx_max'] + 1, size=train_user.shape, device=train_user.device)
    train_item = train_item - data['item_idx_min']
    neg_item = neg_item - data['item_idx_min']

    model.train()
    epoch_loss = 0
    for i, (u, pos_x, neg_x) in enumerate(batcher(train_user, train_item, neg_item, batch_size=batch_size, shuffle=True)):
        optimizer.zero_grad()
        pos_pred = model.forward(u, pos_x)
        neg_pred = model.forward(u, neg_x)
        loss = loss_func(pos_pred, neg_pred)
        loss += 0.5 * reg_l2 * model.l2_sqnorm()
        epoch_loss += loss * batch_size
        loss.backward()
        model.riemannian_grads()
        optimizer.step()
        model.project_embedding()
    print("Average Train Loss: {}".format(epoch_loss / train_user.shape[0]))


def evaluate(model, data, score_func, on='dev'):
    print('Evaluate on', on, 'set',)
    key_user = on + '_user'
    key_item = on + '_item'
    user, item = data[key_user], data[key_item]
    neg_item = torch.randint(low=data['item_idx_min'], high=data['item_idx_max'] + 1, size=user.shape, device=user.device)
    item = item - data['item_idx_min']
    neg_item = neg_item - data['item_idx_min']
    model.eval()
    with torch.no_grad():
        pos_pred = model(user, item)
        neg_pred = model(user, neg_item)
        score = score_func(pos_pred, neg_pred)
        print("-", on, "score: {:.5f}".format(score))
    return score


def parse_args():
    parser = argparse.ArgumentParser(description="Run the model.")
    parser.add_argument('--seed', type=int, default=205,
                        help='Seed for random, numpy, torch and cuda.')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='Whether use cuda.')
    parser.add_argument('--embedding_dim', type=int, default=10,
                        help='Set embedding dimension for the model.')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD', 'Momentum', 'Adagrad'],
                        help='Specify an optimizer type (Adam, Adagrad, SGD, Momentum).')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--lr_poincare', type=float, default=1e-3,
                        help='Learning rate for hyperbolic embeddings')
    parser.add_argument('--reg_l2', type=float, default=0,
                        help='L2 regularization parameter.')
    parser.add_argument('--dataset', default='TX',
                        help='Specify a dataset (ml100k, lastfm, amazon_cloths, ...)')
    parser.add_argument('--use_content', type=int, default=0,
                        help='Whether using content features')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Frequency of evaluating the performance on dev data \
                        (-1: never, 0: at the end, n: every n epochs)')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Whether to plot the hyperbolic embeddings')
    parser.add_argument('--process', default='train',
                        choices=['train', 'eval'],
                        help='Process type: train, evaluate.')
    return parser.parse_args()


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
        for i, (u, pos, tmp) in enumerate(batcher(user, pos_item, user, batch_size=batch_size, shuffle=False)):
            num_user = u.shape[0]
            neg = torch.randint(low=data['item_idx_min'], high=data['item_idx_max'] + 1, size=(num_user, 99), device=user.device)
            pn = torch.cat([pos, neg], dim=1).view((-1, 1))
            u = u.repeat(1, 100).view((-1, 1))
            pn = pn - data['item_idx_min']
            pred = model(u, pn)
            pred = pred.view((num_user, 100))
            score5, score10 = recall_at(pred, target=0)
            epoch_score5 += score5 * num_user
            epoch_score10 += score10 * num_user
        score5 = epoch_score5 / user.shape[0]
        score10 = epoch_score10 / user.shape[0]
        print("-", on, "score5: {:.5f}".format(score5), "score10: {:.5f}".format(score10))
        print()
    return score5, score10


if __name__ == "__main__":
    args = parse_args()

    # fix random seed
    set_random_seed(args.seed)

    # use cuda or not
    if args.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load data
    data, n_fields, n_features = load_data(args.dataset, device=device, use_content=0, use_rating=False, print_info=True)
    num_users = data['item_idx_min']
    num_items = data['item_idx_max'] - data['item_idx_min'] + 1

    # create model
    model = HyperbolicEmbedding(num_users, num_items, embedding_dim=args.embedding_dim)
    model.to(device=device)

    # output dir
    output_dir = "./results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # train
    print("Training start ...")
    start = time.time()
    best_dev_score = train(model, data, output_path=output_path, args=args, loss_func=bpr_loss, score_func=AUC)
    print("Training process took {:.2f} seconds\n".format(time.time() - start))

    # load best model and analysis on test set
    model.load_state_dict(torch.load(output_path))
    test_score = evaluate(model, data, AUC, on='test')
    dev_hit5, dev_hit10 = hit_eval(model, data, on='dev')
    test_hit5, test_hit10 = hit_eval(model, data, on='test')

    # write results
    save_results('./' + args.dataset + '_PMF.txt', args, best_dev_score, test_score, dev_hit5, dev_hit10, test_hit5, test_hit10)
