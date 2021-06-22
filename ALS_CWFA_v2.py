import numpy as np
import torch
import sys
from utils import *
from TT_learning import TT_spectral_learning
torch.set_default_dtype(torch.float64)
from splearn.datasets.base import load_data_sample
from ALS_CWFA import spectral_learning_matrix, tt_to_tensor, test_spice
class ALS_CWFA_v2():
    def __init__(self, **option):
        option_default = {
            'core_list': [],
            'length': 5,
            'rank': 12,
            'input_dim': 3,
            'out_dim': 1,
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

        padding = option['padding']
        self.padding = padding
        self.alpha = torch.normal(0., init_std, [1, rank])
        self.omega = torch.normal(0., init_std, [rank, 1])
        if Cwfa is None:
            if option['resample']:
                core_list = []
                for l in range(0, length):
                    core = torch.normal(0., init_std, [rank, input_dim, rank])/rank
                    core_list.append(core)
                self.core_list = core_list
            else:
                self.core_list = core_list
        else:

            self.alpha = Cwfa.alpha.detach()
            for l in range(0, length ):
                core_list.append(Cwfa.A.detach())

            self.omega = Cwfa.Omega.detach()
            self.core_list = core_list

        self.length  = len(core_list)
        self.rank = core_list[0].shape[-1]
        self.input_dim = core_list[0].shape[1]

    def contract_in_range(self, x, start_index, end_index):
        if end_index == 0:
            merged = torch.ones(x.shape[0], self.alpha.shape[1])
            for i in range(len(merged)):
                merged[i, :] = self.alpha
            return merged.reshape(x.shape[0], 1, merged.shape[1])

        if start_index == 0:
            merged = torch.einsum('ij, jkl -> ikl', self.alpha, self.core_list[start_index])
            merged = torch.einsum('ijk, nj->nik', merged, x[:, start_index, :])
        elif start_index< self.length:
            merged = torch.einsum('ijk, nj->nik', self.core_list[start_index], x[:, start_index, :])
        else:
            merged = torch.ones(x.shape[0], self.omega.shape[0])
            for i in range(len(merged)):
                # print(self.omega.reshape(-1, 1).shape, merged.shape)
                merged[i, :] = self.omega.reshape(-1, )

            return merged.reshape(x.shape[0], 1, merged.shape[1])

        if end_index > self.length:
            for k in range(start_index + 1, end_index-1):
                # print(merged.shape)
                if start_index + 1 == end_index: break
                merged = torch.einsum('nik,kjl, nj -> nil', merged, self.core_list[k], x[:, k, :])

        else:
            for k in range(start_index + 1, end_index):
                # print(merged.shape)
                if start_index + 1 == end_index: break
                merged = torch.einsum('nik,kjl, nj -> nil', merged, self.core_list[k], x[:, k, :])


        if end_index == self.length:
            # print(merged.shape, self.omega.shape)
            merged = torch.einsum('nil, lk-> nik', merged, self.omega)
            # print(merged.shape)
        return merged

    def contract_skip_i(self, x, index_to_skip):
        '''
        :param x: of shape (n, l, d)_
        :return:
        '''
        if index_to_skip == -1:
            merged = self.contract_in_range(x, -1, self.length)
            # print(merged.shape)
            return merged.reshape(x.shape[0], -1)
        if index_to_skip == self.length+1:
            merged = self.contract_in_range(x, 0, self.length+1)
            return merged.reshape(x.shape[0], -1)

        left = self.contract_in_range(x, 0, index_to_skip)
        right = self.contract_in_range(x, index_to_skip+1, self.length)
        # print(left.shape, right.shape)

        merged = torch.einsum('nil, nd, nkm->nildkm', left, x[:, index_to_skip, :], right)

        shape = merged.shape

        return merged.reshape(x.shape[0], -1)

    def evaluate(self, X, Y):
        pred = self.contract_in_range(X, 0, self.length).squeeze()
        pred = pred.reshape(Y.shape)
        # print(pred[:5])
        # print(Y[:5])
        return torch.mean((pred - Y) **2, dim=0), 9999

    def learning(self, x, y, epochs = 100, test_x = None, test_y = None):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).double()
        if not torch.is_tensor(y):
            y = torch.from_numpy(y).double()

        if not torch.is_tensor(test_x):
            test_x = torch.from_numpy(test_x).double()
        if not torch.is_tensor(test_y):
            test_y = torch.from_numpy(test_y).double()
        errors = []
        for k in range(epochs):
            try:
                self.normalize()
                tmp = self.contract_skip_i(x, -1)
                sol, _, _, _ = np.linalg.lstsq(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(),
                                               rcond=None)

                sol = torch.from_numpy(sol).reshape(self.alpha.shape)
                self.alpha = sol

                for curr_skip_index in range(x.shape[1]):
                    tmp = self.contract_skip_i(x, curr_skip_index)
                    # print(tmp.shape, y.shape)
                    sol, _, _, _ = np.linalg.lstsq(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(),
                                                   rcond=None)
                    sol = torch.from_numpy(sol)
                    curr_core = sol.reshape(self.core_list[curr_skip_index].shape)
                    self.core_list[curr_skip_index] = curr_core

                tmp = self.contract_skip_i(x, self.length+1)
                sol, _, _, _ = np.linalg.lstsq(tmp.clone().numpy(), y.reshape(-1, 1).clone().numpy(),
                                               rcond=None)
                sol = torch.from_numpy(sol).reshape(self.omega.shape)
                self.omega = sol

                train_loss = self.evaluate(x, y)
                print(k, train_loss)
                vali_loss = self.evaluate(test_x, test_y)
                print('testing', vali_loss)
                mse, mape = self.evaluate(test_x, test_y)
                errors.append([self.evaluate(x, y), mse, mape])
                # if mse <= 1:
                #     break


                # self.normalize_cores()

            except:
                e = sys.exc_info()
                print('I am here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print("<p>Error: %s</p>" % e)
                break
        return errors
    def extact_weights(self):
        out_cores = []
        for core in self.core_list:
            out_cores.append(core.detach().numpy())
        return self.alpha.detach().numpy(), out_cores, self.omega.detach().numpy()

    def normalize(self):
        alpha = self.alpha.data
        omega = self.omega.data
        norm = torch.norm(alpha, p = 'fro')
        for core in self.core_list:
            tmp = core.data
            norm *= torch.norm(tmp.reshape(tmp.shape[0], -1), p = 'fro')
        norm *= torch.norm(omega, p = 'fro')

        factor = norm ** (1/ (self.length+ 2))

        self.alpha.data = factor * alpha/torch.norm(alpha, p = 'fro')
        for i, core in enumerate(self.core_list):
            tmp = core.data
            self.core_list[i].data = tmp * factor /torch.norm(tmp.reshape(tmp.shape[0], -1), p = 'fro')

        self.omega.data = omega * factor / torch.norm(omega, p = 'fro')

def get_hankel(file_name, l, rank, epochs, version ='classic', exact_length = True, mnne_factor = 1):
    pt = load_data_sample(file_name)
    # from utils import get_hankel_tt_size
    # max_num_negative_examples = get_hankel_tt_size(l, pt.nbL+1, rank) - len()
    if version == 'classic':
        x, y = get_xy_from_splearn(file_name, l = l, nbL=pt.nbL,
                                          mnne_factor = 1, rank_tt=rank)
    elif version == 'factor':
        x, y = get_xy_from_splearn_factor(file_name, l=l, nbL=pt.nbL, lrows=100,
                                          lcolumns=100, exact_length=exact_length,
                                          mnne_factor = 1, rank_tt=rank)

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
    als_learner = ALS_CWFA_v2(**ALS_option)
    errors = als_learner.learning(x, y, epochs=epochs, test_x=x, test_y=y)
    alpha, H, omega = als_learner.extact_weights()
    H[0] = np.einsum('ij, jkl -> ikl', alpha, H[0])
    H[-1] = np.einsum('ijk, kl -> ijl', H[-1], omega)

    return H


if __name__ == '__main__':
    import pickle
    load = False
    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    version = 'classic'
    exact_length = False
    l = 3
    rankHs = [12, 14, 16, 18, 20]
    mnne_factors = [1]
    rank_sps = [12]

    # rankHs = [12]
    # mnne_factors = [100]
    # rank_sps = [12, 13]
    errors = {}
    import pickle

    for rank_H in rankHs:
        for mnne_factor in mnne_factors:
            for rank_sp in rank_sps:
                if rank_sp > rank_H: continue
                if load:
                    with open('ALSv2_H2l', 'rb') as f:
                        H_2l = pickle.load(f)
                    with open('ALSv2_H2l1', 'rb') as f:
                        H_2l1 = pickle.load(f)
                    with open('ALSv2_Hl', 'rb') as f:
                        H_l = pickle.load(f)
                else:
                    H_l = get_hankel(train_file, l, rank = rank_H, epochs=1, version=version, exact_length=exact_length, mnne_factor=mnne_factor)
                    with open('ALSv2_Hl', 'wb') as f:
                        pickle.dump(H_l, f)
                    print(tt_to_tensor(H_l))
                    H_2l = get_hankel(train_file, 2*l, rank=rank_H, epochs=10, version=version, exact_length=exact_length, mnne_factor=mnne_factor)
                    with open('ALSv2_H2l', 'wb') as f:
                        pickle.dump(H_2l, f)
                    # print(tt_to_tensor(H_2l))
                    H_2l1 = get_hankel(train_file, 2*l+1, rank=rank_H, epochs=10, version=version, exact_length=exact_length, mnne_factor=mnne_factor)
                    with open('ALSv2_H2l1', 'wb') as f:
                        pickle.dump(H_2l1, f)
                    # print(tt_to_tensor(H_2l1))




                model = spectral_learning_matrix(tt_to_tensor(H_l), tt_to_tensor(H_2l), tt_to_tensor(H_2l1), rank=12)
                # model = TT_spectral_learning(H_2l, H_2l1, H_l)
                if version == 'factor':
                    model.factor_to_classic()
                error = test_spice(model, test_file, solution_file, sp=exact_length)
                key = str([rank_H, mnne_factor, rank_sp])
                if key not in errors:
                    errors[key] = {
                        'rank_H': rank_H,
                        'rank_sp': rank_sp,
                        'negative_example factor': mnne_factor,
                        'error': error
                    }
                    print(errors)
                    with open('als_cwfa_v2_result2', 'wb') as f:
                        pickle.dump(errors, f)
