import numpy as np
import torch
from TT_learning import TT_spectral_learning
from CWFA import CWFA

import splearn
import splearn.datasets.base as spb
import tensorly as tl
import sys
from utils import *
torch.set_default_dtype(torch.float64)
from splearn.datasets.base import load_data_sample
from sklearn.linear_model import Ridge
from sklearn import linear_model
from LinRNN import LinRNN
import copy
from tensorly.decomposition import tensor_train

def run_ridge(x, y, alpha = 1.):
    clf = Ridge(alpha=alpha, fit_intercept = False)
    clf.fit(x, y)
    return clf.coef_

def run_lasso(x, y, alpha = 1.):
    clf = linear_model.Lasso(alpha=alpha, fit_intercept = False)
    clf.fit(x, y)
    return clf.coef_


class ALS_CWFA():
    def __init__(self, **option):
        option_default = {
            'core_list': [],
            'length': 5,
            'rank': 4,
            'input_dim': 3,
            'out_dim': 2,
            'init_std': 0.1,
            'cwfa': None,
            'resample': True,
            'padding': True
        }

        option = {**option_default, **option}
        core_list = option['core_list']
        length = option['length']
        rank = option['rank']
        input_dim = option['input_dim']
        out_dim = option['out_dim']
        init_std = option['init_std']
        total_num = rank * input_dim * rank *length
        Cwfa = option['cwfa']
        torch.manual_seed(0)
        print('1', Cwfa, Cwfa is None, len(core_list))

        padding = option['padding']
        self.padding = padding

        if Cwfa is None:
            if option['resample']:
                core_list = []
                if length == 1:
                    core = torch.normal(0., init_std, [1, input_dim, out_dim])
                else:
                    core = torch.normal(0., init_std, [1, input_dim, rank])
                core_list.append(core)
                print(Cwfa)
                for l in range(1, length-1):
                    core = torch.normal(0., init_std, [rank, input_dim, rank])/rank
                    # for m in range(input_dim):
                    #     core[:, m, :] += torch.eye(rank, rank)
                    # if padding:
                    #     core[:, -1, :] = torch.eye(rank)
                    core_list.append(core)
                core = torch.normal(0., init_std, [rank, input_dim, out_dim])
                core = core/rank
                # core = core/total_num
                core_list.append(core)
                self.core_list = core_list
            else:
                self.core_list = core_list
        else:
            print('got here')
            core = torch.einsum('ni, ijk->njk', Cwfa.alpha.detach().reshape(1, -1), Cwfa.A.detach())
            core_list.append(core)
            for l in range(1, length - 1):
                core_list.append(Cwfa.A.detach())

            core = torch.einsum('ijk, km -> ijm', Cwfa.A.detach(), Cwfa.Omega.detach().reshape(-1, 1))
            core_list.append(core)
            self.core_list = core_list


        self.length  = len(core_list)
        self.rank = core_list[0].shape[-1]
        self.input_dim = core_list[0].shape[1]

    def contract_in_range(self, x, start_index, end_index):
        # print( self.core_list[start_index].shape,  x[:, start_index, :].shape)
        merged = torch.einsum('ijk, nj->nik', self.core_list[start_index], x[:, start_index, :])
        # print('0', merged[0, 0, 0])
        for k in range(start_index+1,end_index):
            merged = torch.einsum('nik,kjl, nj -> nil', merged, self.core_list[k], x[:, k, :])
            # print(k ,merged[0, 0, 0])
        return merged

    def normalize_cores(self):
        tt_norm = 1.
        for k in range(self.length):
            tt_norm *= torch.norm(self.core_list[k].reshape(self.core_list[k].shape[0], -1), p = 'fro')
        ave_norm = tt_norm ** (1 / self.length)

        for k in range(self.length):
            self.core_list[k] /= torch.norm(self.core_list[k].reshape(self.core_list[k].shape[0], -1), p = 'fro')
            self.core_list[k] *= ave_norm




    def contract_skip_i(self, x, index_to_skip):
        '''
        :param x: of shape (n, l, d)_
        :return:
        '''
        if index_to_skip == 0:
            merged = self.contract_in_range(x, 1, self.length)
            merged = torch.einsum('nkm, nd -> ndkm', merged, x[:, 0, :])
            return merged.reshape(merged.shape[0] * merged.shape[-1], -1)
        if index_to_skip >= self.length-1:
            merged = self.contract_in_range(x, 0, self.length - 1)
            merged = torch.einsum('nil, nd -> nild', merged, x[:, -1, :])
            return merged.reshape(merged.shape[0], -1)

        left = self.contract_in_range(x, 0, index_to_skip)
        right = self.contract_in_range(x, index_to_skip+1, self.length)
        merged = torch.einsum('nil, nd, nkm->nildkm', left, x[:, index_to_skip, :], right)

        shape = merged.shape

        return merged.reshape(shape[0] * shape[-1], -1)

    def evaluate(self, X, Y):
        pred = self.contract_in_range(X, 0, self.length).squeeze()
        pred = pred.reshape(Y.shape)
        # print(pred[:5])
        # print(Y[:5])
        return torch.mean((pred - Y) **2, dim=0), 9999


    def ALS_learning(self, x, y, epochs = 100, test_x = None, test_y = None, beta = 1e-30):
        '''
        :param x: of shape (n, l, d)_
        :return:
        '''
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).double()
        if not torch.is_tensor(y):
            y = torch.from_numpy(y).double()

        if not torch.is_tensor(test_x):
            test_x = torch.from_numpy(test_x).double()
        if not torch.is_tensor(test_y):
            test_y = torch.from_numpy(test_y).double()
        # print('initial error', self.evaluate(x, y))
        errors = []
        get_out = False
        count_down = 30
        for k in range(epochs):
            try:
                # print('first', self.core_list[0].shape)
                for curr_skip_index in range(x.shape[1]):
                    tmp = self.contract_skip_i(x, curr_skip_index)
                    # print('original:', self.core_list[curr_skip_index], self.core_list[curr_skip_index].shape)
                    if curr_skip_index == x.shape[1] - 1:
                        sol, _, _, _ = np.linalg.lstsq(tmp.clone().numpy(), y.clone().numpy(), rcond=None)
                        # sol = run_ridge(tmp.clone().numpy(), y.clone().numpy(), alpha=beta)
                        # sol = run_lasso(tmp.clone().numpy(), y.clone().numpy(), alpha=beta)
                        sol = torch.from_numpy(sol)
                        curr_core = sol.reshape(self.core_list[curr_skip_index].shape)

                    else:
                        sol, _, _, _ = np.linalg.lstsq(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(), rcond=None)

                        # print(sol)

                        # sol = run_ridge(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(), alpha=beta)
                        # sol = run_lasso(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(), alpha=beta)
                        sol = torch.from_numpy(sol)
                        curr_core = sol.reshape(self.core_list[curr_skip_index].shape)
                    self.core_list[curr_skip_index] = curr_core

                train_loss = self.evaluate(x, y)
                print(k, train_loss)
                vali_loss = self.evaluate(test_x, test_y)
                print('testing', vali_loss)
                mse, mape = self.evaluate(test_x, test_y)
                errors.append([self.evaluate(x, y), mse, mape])

                if mape <=1e-3 and not get_out:
                    get_out = True
                if get_out:
                    count_down -= 1
                    if count_down <=0: break
                self.normalize_cores()

            except:
                # e = sys.exc_info()
                print('I am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # print("<p>Error: %s</p>" % e)
                break
        return errors

    def extact_weights(self):
        out_cores = []
        for core in self.core_list:
            out_cores.append(core.detach().numpy())
        return out_cores







def MAPE(pred, y):
    err = pred - y
    err = np.divide(err, y)
    # print(pred[:3], y[:3])
    return np.mean(np.abs(err))

def get_counts(x):
    dict = {}
    for i in range(len(x)):
        key = str(x[i])
        if key in dict:
            dict[key] += int(1)
        else:
            dict[key] = int(1)
    # for key in dict.keys():
    #     dict[key]/=len(x)
    return dict

# def pad(x, symbol, max_length, mode = 'random'):
#
#     if np.isscalar(x[0]):
#         # print('scalr')
#         # print(max_length - len(x))
#         for j in range(max_length - len(x)):
#             if mode == 'random':
#                 pos = np.random.randint(0, len(x) + 1, size = 1)[0]
#                 # print(pos)
#                 x = np.insert(x, pos, symbol)
#             elif mode == 'append':
#                 x = np.insert(x, len(x), symbol)
#             elif mode == 'splearn':
#                 if j == 0:
#                     x = np.insert(x, 0, symbol)
#                 else:
#                     x = np.insert(x, len(x), symbol)
#
#     else:
#         # print('array')
#         for i in range(len(x)):
#             # print(max_length - len(x[i]))
#             for j in range(max_length - len(x[i])):
#                 if mode == 'random':
#                     pos = np.random.randint(0, len(x[i]) + 1, size = 1)[0]
#                     # print(symbol)
#                     x[i] = np.insert(x[i], pos, symbol)
#                     # print(x[i])
#                 elif mode == 'append':
#                     x[i] = np.insert(x[i], len(x[i]), symbol)
#                 elif mode == 'splearn':
#                     if j == 0:
#                         x[i] = np.insert(x[i], 0, symbol)
#                     else:
#                         x[i] = np.insert(x[i], len(x[i]), symbol)
#
#     return x
#
#
def get_spice_data(file_name):
    #, no_padding = False, pad_mode = 'random'):
    pt = load_data_sample(file_name)
    data = pt.data
    max_len = data.shape[1]

    dict = get_counts(data)
    y = []
    for i in range(len(data)):
        y.append(dict[str(data[i])])

    new_data = []
    for i in range(len(data)):
        flag = True
        for j in range(len(data[i])):
            if data[i][j] == -1:
                # print(1, data[i][:j + 1], j)
                new_data.append(data[i][:j])
                # print(2, new_data[-1])
                flag = False
                break
        if flag:
            new_data.append(data[i])
    return new_data, y
    # if no_padding:
    #     return new_data, y
    # else:
    #     # print('got here')
    #     new_data = pad(new_data, -1, max_len, pad_mode)
    #     return new_data, y
#
# def filter_length(x, y, l):
#
#     filtered_x = []
#     filtered_y = []
#     for i in range(len(x)):
#         for j in range(len(x[i])):
#             if x[i][j] == -1:
#                 break
#         if j <= l:
#             filtered_x.append(x[i][:l])
#             filtered_y.append(y[i])
#     # filtered_x = np.asarray(filtered_x)[:, :l]
#     # filtered_y = np.asarray(filtered_y)
#     return filtered_x, filtered_y
#
# def add_negative_examples(x, y, max_length, num_negative_examples = 10):
#     dict = {}
#     for i in range(len(x)):
#         key = str(x[i])
#         if key not in dict:
#             dict[key] = 1
#     all_prevs = []
#     all_suffs = []
#     for i in range(len(x)):
#         # for j in range(1, len(x[i])):
#         #     # print(x[i][j])
#         #     if x[i][j] == -1: break
#         for k in range(len(x[i])):
#             prev = x[i][:k]
#             suff = x[i][k:]
#             all_prevs.append(prev)
#             all_suffs.append(suff)
#     count = 0
#     visitation = 0
#
#     # new_x = []
#     # for tmp in x:
#     #     new_x.append(tmp)
#     while True:
#         if count >= num_negative_examples: break
#         # visitation += 1
#         # if visitation >= 10*num_negative_examples: break
#         k = np.random.random_integers(0, len(all_prevs)-1, size=1)[0]
#         q = np.random.random_integers(0, len(all_suffs)-1, size=1)[0]
#         if k == q: continue
#         curr = np.concatenate((all_prevs[k], all_suffs[q]))
#         if len(curr) > max_length: continue
#         # curr = pad(curr, symbol = -1, max_length = x.shape[1], mode = pad_mode).reshape(1, -1)
#         key = str(curr)
#         # print(curr)
#         if key in dict: continue
#         # new_x = np.append(x, list(curr), axis=0)
#         x.append(curr)
#         y = np.append(y, 0.)
#         # print(x)
#         count += 1
#
#     return x, y
#
# def add_negative_examples_v2(x, y, max_length, nbL = 4, max_num_negative_examples = 10):
#     all_trajs = get_traj_up_to_L([[]], nbL, max_length)
#     dict = {}
#     for tmp in x:
#         key = str(tmp)
#         dict[key] = 1
#     for traj in all_trajs:
#         key = str(traj)
#         if key not in dict:
#             dict[key] = 1
#             x.append(traj)
#             y = np.append(y, 0.)
#     return x, y
#
#
#
#
#
# def generate_discrete_data(n, l, d):
#     x = np.random.random_integers(0, d-1, size=[n, l])
#     x = one_hot_no_pad(x, d)
#     return x

def spectral_learning_matrix(H_l, H_2l, H_2l1, rank):
    if H_2l.ndim % 2 == 0: # scalar outputs
        out_dim = 1
        l = H_l.ndim
    else:
        out_dim = H_2l.shape[-1]
        l = H_l.ndim - 1
    # print(l)
    d = H_l.shape[0]
    print(H_2l.shape, H_2l1.shape, H_l.shape)
    H2l = set_to_zero(H_2l.reshape([d ** l, d ** l * out_dim]))
    H_2l1 = set_to_zero(H_2l1.reshape([d ** l, d, d ** l * out_dim]))
    H_l = set_to_zero(H_l.ravel())

    # H2l += np.random.normal(0, 1, H2l.shape)
    # H_2l1 += np.random.normal(0, 1, H_2l1.shape)
    # H_l += np.random.normal(0, 1, H_l.shape)

    # H2l = set_to_zero(H_2l)
    U, s, V = np.linalg.svd(H2l)
    U = U[:, :rank]
    V = V[:rank, :]
    s = s[:rank]

    # print(s)

    Pinv = np.linalg.pinv(U @ np.diag(s))
    Sinv = np.linalg.pinv(V)


    print('H2l1', H_2l1)
    # H_2l1 = set_to_zero(H_2l1)

    A = np.tensordot(Pinv, H_2l1, axes=(1, 0))
    A = np.tensordot(A, Sinv, axes=[-1, 0])
    # print(A)


    # H_l = H_l
    if out_dim == 1:
        omega = Pinv.dot(H_l.ravel())
    else:
        omega = (Pinv.dot((H_l.reshape([d**l,out_dim]))))
    alpha = Sinv.T.dot(H_l.ravel())
    model = LinRNN(alpha, A, omega)
    print('A', A[:, -1, :].diagonal())
    return model

def filter_repetition(x, y):
    dict = {}
    new_x = []
    new_y = []
    for i in range(len(x)):
        if str(x[i]) not in dict:
            dict[str(x[i])] = 1
            new_x.append(x[i])
            new_y.append(y[i])
        else:
            continue
    return new_x, new_y

def convert_int(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = int(x[i][j])
    return x
#
# def get_hankel_spice(file_name, als_rank, length, epochs = 1000, pad_mode = 'splearn'):
#     np.random.seed(10)
#     x, y = get_spice_data(file_name)
#     # print(1, x)
#     x = convert_int(x)
#     x, y = filter_length(x, y, l=length)
#     x, y = filter_repetition(x, y)
#     x = pad(x, -1, length, pad_mode)
#     # print(x)
#     x, y = add_negative_examples(np.asarray(x), np.asarray(y), max_length=length, num_negative_examples = 1000)
#     x, y = filter_repetition(x, y)
#     x = convert_int(x)
#     # print(1, x)
#     # print(1, y)
#     pt = load_data_sample(file_name)
#     x = one_hot(x, pt.nbL)
#
#     # print(x.shape)
#     all_indices = np.arange(len(x))
#     all_indices = np.random.permutation(all_indices)
#     sep = int(0.8 * len(x))
#     train_indices = all_indices
#     test_indices = all_indices[sep:]
#
#     train_x = np.asarray(x)[train_indices]
#     train_y = np.asarray(y)[train_indices]
#     test_x = np.asarray(x)[test_indices]
#     test_y = np.asarray(y)[test_indices]
#
#     ALS_option = {
#         'core_list': [],
#         'length': length,
#         'rank': als_rank,
#         'input_dim': pt.nbL+1,
#         'out_dim': 1,
#         'init_std': 0.0001,
#         'cwfa': None
#     }
#     als_learner = ALS_CWFA(**ALS_option)
#     errors = als_learner.ALS_learning(train_x, train_y, epochs=epochs, test_x=test_x, test_y=test_y)
#     print(train_x)
#     import pickle
#     # n_l_d_k
#     file_name = 'als_' + file_name + '.error'
#     with open(file_name, 'wb') as f:
#         pickle.dump(errors, f)
#     return als_learner.extact_weights()

def tt_to_tensor(H):
    return tl.tt_tensor.mps_to_tensor(H)

def compute_perplexity(pred, y):
    tmp = 0.
    pred = np.abs(pred)
    for i in range(len(pred)):
        tmp += y[i]*np.log2(pred[i])

    return 2**(-tmp)

def normalize_probability(pred):
    return pred/np.sum(pred)


def test_spice(mdl, test_file, test_solution, sp= False):
    x, y = get_spice_data(test_file)
    pt = load_data_sample(test_file)
    pred = []
    if sp:
        x = one_hot(x, pt.nbL - 1)
    else:
        x = one_hot(x, pt.nbL)
    for o in x:
        # print(o)
        # print(mdl.A.shape)
        pred.append(mdl.predict(o))
    pred = np.abs(np.array(pred).reshape(len(y),))
    pred = normalize_probability(pred)
    # print(pred.shape)
    y = np.genfromtxt(test_solution, delimiter=',')[1:].reshape(len(pred),)
    print(pred[:5], y[:5])
    return compute_perplexity(pred, y)

def set_to_zero(h):
    if h.ndim == 3:
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                for k in range(h.shape[2]):
                    if h[i][j][k] < 1:
                        h[i][j][k] = 0
    elif h.ndim == 2:
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i][j] < 1:
                    h[i][j] = 0
    elif h.ndim == 1:
        for i in range(h.shape[0]):
            if h[i] < 1:
                h[i] = 0

    return h

def thresholding(tt, rank = 12):
    H = tt_to_tensor(tt)
    # if len(tt) == 4:
    #     print('HHHH', len(tt), H.shape, H[])
    tt = tl.decomposition.tensor_train(H, rank=rank)
    return tt_to_tensor(tt)

# def get_xy_from_splearn(train_file, l, nbL, rank = 2, lrows=1, lcolumns=1):
#     train = spb.load_data_sample(train_file)
#     est = splearn.Spectral(rank=rank, lrows=lrows, lcolumns=lcolumns)
#     est.fit(train.data)
#     train_dicts = est.polulate_dictionnaries(train.data)
#     x = []
#     y = []
#     for key in train_dicts.sample.keys():
#         x.append(list(key))
#         y.append(train_dicts.sample[key])
#     return x, y

# def get_hankel_from_xy(x, y, l, rank, nbL, epochs):
#     selected_indices = []
#     for i in range(len(x)):
#         if len(x[i]) <= l:
#             selected_indices.append(i)
#     x = [x[index] for index in selected_indices]
#     y = np.asarray([y[index] for index in selected_indices])
#     x, y = add_negative_examples_v2(x, y, max_length= l, nbL = nbL, max_num_negative_examples=100000)
#     new_x = np.asarray(pad(copy.deepcopy(x), -1, l, mode='random'))
#     new_y = np.asarray(copy.deepcopy(y))
#     # print(new_x.shape, y.shape)
#     for i in range(10):
#         tmp = np.asarray(pad(copy.deepcopy(x), -1, l, mode='random'))
#         # print(i, tmp.shape)
#         new_x = np.concatenate((new_x, tmp), axis = 0)
#         new_y = np.concatenate((new_y, copy.deepcopy(y)), axis=0)
#
#     x = one_hot(new_x, nbL-1)
#     x = np.asarray(x)
#
#     y = new_y
#     x, y = filter_repetition(x, y)
#     x = np.asarray(x)
#     y = np.asarray(y)
#     idx = np.arange(len(x))
#     idx = np.random.permutation(idx)
#     x = x[idx]
#     y = y[idx]
#     print(np.asarray(x).shape)
#     print(x[-1])
#
#
#     # print(x.shape)
#     # print(y.shape)
#     L = x.shape[1]
#     ALS_option = {
#         'core_list': [],
#         'length': L,
#         'rank': rank,
#         'input_dim': nbL,
#         'out_dim': 1,
#         'init_std': 0.0001,
#         'cwfa': None
#     }
#     # print(ALS_option)
#     als_learner = ALS_CWFA(**ALS_option)
#     errors = als_learner.ALS_learning(x, y, epochs=epochs, test_x=x, test_y=y, beta = 1e-1)
#     H  = als_learner.extact_weights()
#     # print(H)
#     # print(tt_to_tensor(H))
#
#
#
#     return H

def get_hankel(file_name, l, rank, epochs, version ='classic', exact_length = True):
    pt = load_data_sample(file_name)
    if version == 'classic':
        x, y = get_xy_from_splearn(file_name, l = l, nbL=pt.nbL)
    elif version == 'factor':
        x, y = get_xy_from_splearn_factor(file_name, l=l, nbL=pt.nbL, lrows=100, lcolumns=100, exact_length=exact_length)

    ALS_option = {
        'core_list': [],
        'length': x.shape[1],
        'rank': rank,
        'input_dim': x.shape[2],
        'out_dim': 1,
        'init_std': 0.0001,
        'cwfa': None
    }
    # print(ALS_option)
    als_learner = ALS_CWFA(**ALS_option)
    errors = als_learner.ALS_learning(x, y, epochs=epochs, test_x=x, test_y=y)
    H = als_learner.extact_weights()

    return H

def spectral_learning_sp_learn(H_l, H_2l, H_2l1, rank):
    # H_l = set_to_zero(H_l)
    # H_2l = set_to_zero(H_2l)

    U, s, V = np.linalg.svd(H_2l)
    # idx = np.argsort(s)[::-1]
    U = U[:, :rank]
    V = V[:rank, :]
    s = s[:rank]


    Pinv = np.diag(1. / s).dot(U.T)
    Sinv = V.T
    # print(Pinv)

    # H_2l1 = H_2l1.reshape([d ** l, d, d ** l * out_dim])
    H2l1 = np.zeros((H_2l.shape[0], len(H_2l1), H_2l.shape[1]))
    for i in range(len(H_2l1)):
        H2l1[:, i, :] = (H_2l1[i])
    # H_2l1 = np.asarray(H_2l1)


    # H2l1 = set_to_zero(H_2l1)
    print(H2l1.shape)
    A = np.tensordot(Pinv, H2l1, axes=(1, 0))
    A = np.tensordot(A, Sinv, axes=[2, 0])
    # print(A)
    h_l = H_l.ravel()
    print(Pinv.shape, Sinv.shape, h_l.shape)
    omega = Pinv.dot(H_2l[:, 0])
    alpha = Sinv.T.dot(H_2l[0, :])
    print(alpha.shape, omega.shape, A.shape)
    model = LinRNN(alpha, A, omega)

    return model




if __name__ == '__main__':
    import pickle
    load = True
    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    version = 'classic'
    exact_length = False
    l = 3
    if load:
        with open('ALS_H2l', 'rb') as f:
            H_2l = pickle.load(f)
        with open('ALS_H2l1', 'rb') as f:
            H_2l1 = pickle.load(f)
        with open('ALS_Hl', 'rb') as f:
            H_l = pickle.load(f)
    else:
        H_l = get_hankel(train_file, l, rank = 15, epochs=100, version=version, exact_length=exact_length)
        print(tt_to_tensor(H_l))
        H_2l = get_hankel(train_file, 2*l, rank=15, epochs=1000, version=version, exact_length=exact_length)
        print(tt_to_tensor(H_2l))
        H_2l1 = get_hankel(train_file, 2*l+1, rank=15, epochs=1000, version=version, exact_length=exact_length)
        print(tt_to_tensor(H_2l1))

        with open('ALS_H2l', 'wb') as f:
            pickle.dump(H_2l, f)
        with open('ALS_H2l1', 'wb') as f:
            pickle.dump(H_2l1, f)
        with open('ALS_Hl', 'wb') as f:
            pickle.dump(H_l, f)
    # print(tt_to_tensor(H_l))

    # train = spb.load_data_sample(train_file)
    # test = spb.load_data_sample(test_file)
    # est = splearn.Spectral(rank=12, lrows=3, lcolumns=3)
    # est.fit(train.data)
    # lhankel = est._hankel.lhankel
    # lhankelA = []
    # for h in lhankel:
    #     h = h.A
    #     for i in range(len(h)):
    #         for j in range(len(h[i])):
    #             if h[i][j] == 0:
    #                 h[i][j] += 0
    #     lhankelA.append(h)
    #
    #
    # H_2l = lhankelA[0]
    # H_2l1 = [lhankelA[i] for i in range(1, len(lhankelA))]
    # H_l = lhankelA[0][0, :]
    # # print(H_2l)
    #
    # H_2l_true = copy.deepcopy(H_2l)
    # H_2l += np.random.normal(0, 10, H_2l.shape)
    # # print(np.mean((H_2l - H_2l_true)**2))
    # #H_l += np.random.normal(0, 0.1, H_l.shape)
    # for i  in range(len(H_2l1)):
    #     H_2l1[i] += np.random.normal(0, 11, H_2l1[i].shape)



    # model = TT_spectral_learning(H_2l, H_2l1, H_l)
    # model = spectral_learning_sp_learn(H_l, H_2l, H_2l1, rank=12)

    # thresholding(tt, rank=12)

    model = spectral_learning_matrix(tt_to_tensor(H_l), tt_to_tensor(H_2l), tt_to_tensor(H_2l1), rank=12)
    if version == 'factor':
        model.factor_to_classic()
    # H_l = tt_to_tensor(H_l)
    # print(H_l.reshape(H_l.shape[0], -1))

    # H_2l = tt_to_tensor(H_2l)
    # print(H_2l.reshape(H_2l.shape[0], -1))
    #
    # print(H_l.reshape(H_l.shape[0], -1))
    # nomralizer = abs(max(model.alpha))
    # model.alpha/=nomralizer
    # model.Omega *= nomralizer
    #print(model.alpha, model.A, model.Omega)
    error = test_spice(model, test_file, solution_file, sp = exact_length)
    print(error)
    # epochs = [10, 100, 100]
    # pad_mode = 'random'
    # H_l = get_hankel_spice(train_file, 12, l, epochs[0], pad_mode)
    # # print(H_l)
    # print(tt_to_tensor(H_l))
    # H_2l = get_hankel_spice(train_file, 12, 2*l, epochs[1], pad_mode)
    # # print(tt_to_tensor(H_2l))
    # H_2l1 = get_hankel_spice(train_file, 12, 2*l+1, epochs[2], pad_mode)
    # # print(tt_to_tensor(H_2l1))
    # model = spectral_learning_matrix(tt_to_tensor(H_l), tt_to_tensor(H_2l), tt_to_tensor(H_2l1), rank = 12)
    # error = test_spice(model, test_file, solution_file)
    # print(error)


    # Ns = [10000]
    # ls = [x.shape[1]]
    # ds = [x.shape[-1]]
    # rs = [15]
    # # Ns = [100000]
    # # ls = [10]
    # # ds = [20]
    # # rs = [4]
    # epochs = 1000
    #
    # param_list = []
    # for n in Ns:
    #     for l in ls:
    #         for d in ds:
    #             for r in rs:
    #                 param_list.append([n, l, d, r])
    #
    # for param in param_list:
    #     n = param[0]
    #     l = param[1]
    #     d = param[2]
    #     r = param[3]
    #     option_default = {
    #             'A': None,
    #             'alpha': None,
    #             'Omega': None,
    #             'example_WFA': True,
    #             'rank': 3,
    #             'device': 'cpu',
    #             'input_dim': d,
    #             'init_std': 1,
    #             'out_dim': 1
    #     }
    #
    #     d = option_default['input_dim']
    #     cwfa = CWFA(**option_default)
    #     # train_x = generate_discrete_data(n, l, d)
    #     # print(train_x.shape)
    #     # train_x = np.random.rand(n, l, d)
    #     # train_y = cwfa(train_x).detach().numpy()
    #
    #     ALS_option = {
    #         'core_list': [],
    #         'length': l,
    #         'rank':r,
    #         'input_dim': d,
    #         'out_dim': 1,
    #         'init_std': 0.01,
    #         'cwfa': None
    #     }
    #     als_learner = ALS_CWFA(**ALS_option)
    #     # test_x = np.random.rand(1000, l, d)
    #     # test_x = generate_discrete_data(1000, l, d)
    #     # test_y = cwfa(test_x).detach().numpy()
    #     errors = als_learner.ALS_learning(train_x, train_y, epochs=epochs, test_x = test_x, test_y = test_y)
    #     import pickle
    #     # n_l_d_k
    #     file_name = 'als_'+str(n)+'_'+str(l)+'_'+str(d)+'_'+str(ALS_option['rank'])+'.error7'
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(errors, f)






