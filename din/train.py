import time
import random
import sys, os
import numpy as np
import argparse
import torch
sys.path.append(os.getcwd())
from din.data_iterator import DataIterator
from din.model import DeepInterestNetwork
from din.utils import *


EMBEDDING_DIM = 12
HIDDEN_DIM = [108,200,80,2]
ATTENTION_SIZE = EMBEDDING_DIM * 2
best_auc = 0.0
device = "cuda" if torch.cuda.is_available() else "cpu"


def transform(data):
    return torch.from_numpy(data).to(device)


def prepare_data(input, target, maxlen=None, return_neg=False):
    # x: a list of sentences
    lengths_x = [len(s[4]) for s in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
    noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])
    cats = np.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x), noclk_mid_his, noclk_cat_his

    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)


def eval(test_data, model, model_path):
    test_data.reset()

    loss_sum = 0.
    accuracy_sum = 0.
    nums = 0
    stored_arr = []
    for _ in range(100):
        src, tgt = test_data.next()
        # for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        uids = transform(uids)
        mids = transform(mids)
        cats = transform(cats)
        mid_his = transform(mid_his)
        cat_his = transform(cat_his)
        mid_mask = transform(mid_mask)
        noclk_mids = transform(noclk_mids)
        noclk_cats = transform(noclk_cats)
        target = transform(target)

        prob = model(uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats)
        
        loss = - torch.mean(torch.log(prob) * target)
        # acc = torch.mean(torch.round(prob) == target)
        acc = torch.sum(torch.round(prob) * target) / target.shape[0]
        loss_sum += loss
        # aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p ,t in zip(prob_1, target_1):
            stored_arr.append([p, t])
    
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums

    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        torch.save({'model_state_dict': model.state_dict()}, model_path)
    return test_auc, loss_sum, accuracy_sum


def train_one_epoch(epoch, model, train_data, test_data, optimizer,
                    maxlen, test_iter, save_iter, best_model_path, model_path):
    train_data.reset()
    iter = 0
    loss_sum = 0.0
    accuracy_sum = 0.
    for _ in range(8000):
        optimizer.zero_grad()
        
        src, tgt = train_data.next()
        # (B,), (B), (B), (B, 100), (B, 100), (B, 100), (B, 2), (B), (128, 100, 5), (128, 100, 5)
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, maxlen, return_neg=True)
        uids = transform(uids)
        mids = transform(mids)
        cats = transform(cats)
        mid_his = transform(mid_his)
        cat_his = transform(cat_his)
        mid_mask = transform(mid_mask)
        noclk_mids = transform(noclk_mids)
        noclk_cats = transform(noclk_cats)
        target = transform(target)

        y_hat = model(uids, mids, cats, mid_his, cat_his, mid_mask, noclk_mids, noclk_cats)
        y_hat = y_hat + 1e-8

        loss = - torch.mean(torch.log(y_hat) * target)
        # acc = torch.mean(torch.round(y_hat) == target)
        acc = torch.sum(torch.round(y_hat) * target) / target.shape[0]

        loss_sum += loss
        accuracy_sum += acc

        loss.backward()
        optimizer.step()
        iter += 1

        if (iter % test_iter) == 0:
            print('[epoch: %d/iter: %d] ----> train_loss: %.4f ---- train_accuracy: %.4f' % \
                                    (epoch, iter, loss_sum / test_iter, accuracy_sum / test_iter))
            test_auc, test_loss, test_accuracy = eval(test_data, model, best_model_path)
            print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % (test_auc, test_loss.data, test_accuracy.data))
            loss_sum = 0.0
            accuracy_sum = 0.0

        if (iter % save_iter) == 0:
            # print('save model iter: %d' %(iter))
            torch.save({
                'EPOCH': epoch,
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"{model_path}_ep{epoch}_{iter}")

    return model, optimizer


def train(
        train_file = "data/local_train_splitByUser",
        test_file = "data/local_test_splitByUser",
        uid_voc = "data/uid_voc.pkl",
        mid_voc = "data/mid_voc.pkl",
        cat_voc = "data/cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        test_iter = 500,
        save_iter = 1000,
        model_type = 'DIN',
        seed = 2,
        epochs = 5
    ):
    out_dir1 = "output"
    out_dir2 = "best_model"
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    model_path = f"{out_dir1}/ckpt_noshuff{model_type}{str(seed)}"
    best_model_path = f"{out_dir2}/ckpt_noshuff{model_type}{str(seed)}"

    train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)
    test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
    n_uid, n_mid, n_cat = train_data.get_n() #uid: 543060, mid: 367983, cat: 1601

    if model_type == 'DIN':
        model = DeepInterestNetwork(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_DIM)
    else:
        print("Invalid model_type : %s", model_type)
        model = None
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {total_params}")

    model.to(device)

    print('test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f' % eval(test_data, model, best_model_path))

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0001)
    for epoch in range(1,epochs+1):
        model, optimizer = train_one_epoch(epoch, model, train_data, test_data, optimizer,
                                           maxlen, test_iter, save_iter, best_model_path, model_path)


def test(
        test_file = "data/local_test_splitByUser",
        uid_voc = "data/uid_voc.pkl",
        mid_voc = "data/mid_voc.pkl",
        cat_voc = "data/cat_voc.pkl",
        batch_size = 128,
        maxlen = 100,
        model_type = 'DIN',
	    seed = 2,
        model_path = ""
    ):
    if model_path == "" or model_path is None:
        model_path = "best_model/ckpt_noshuff" + model_type + str(seed)

    # train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
    test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
    n_uid, n_mid, n_cat = test_data.get_n()
    if model_type == 'DIN':
        model = DeepInterestNetwork(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_DIM)
    else:
        print ("Invalid model_type : %s", model_type)
        return
    model.to(device)

    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint)
    
    print('test_auc: %.4f ----test_loss: %.4f ---- test_accuracy: %.4f' % eval(test_data, model, model_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CTR")
    parser.add_argument("--mode", default="train", help="train/test", type=str)
    parser.add_argument("--model", default="DIN", help="DIN/DIEN/DEBUG", type=str)
    parser.add_argument("--ep", default=5, help="total training epochs", type=int)
    parser.add_argument("--model_path", default="", help="model ckpt to test", type=str)
    args = parser.parse_args()

    SEED = 4
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    # Other randomization-related settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    if args.mode == 'train':
        train(model_type=args.model, seed=SEED, epochs=args.ep)
    elif args.mode == 'test':
        test(model_type=args.model, model_path=args.model_path, seed=SEED)
    else:
        print('do nothing...')

