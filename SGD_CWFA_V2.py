import numpy as np
import torch
from torch import nn
from gradient_descent import *
from Dataset import Dataset
import pickle
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
# from TT_learning import TT_spectral_learning
# from CWFA import CWFA
from tensorly import tt_to_tensor
from ALS_CWFA import spectral_learning_matrix, test_spice
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

class SGD_Hankel_v2(nn.Module):
    def __init__(self, **option):
        super().__init__()
        option_default = {
            'length': 5,
            'rank': 4,
            'input_dim': 3,
            'out_dim': 2,
            'init_std': 0.0,
            'padding': True
        }

        option = {**option_default, **option}
        self.length = option['length']
        self.rank = option['rank']
        self.input_dim = option['input_dim']
        self.out_dim = option['out_dim']
        self.init_std = option['init_std']
        torch.manual_seed(0)
        padding = option['padding']
        self.padding = padding
        print(self.padding)
        if not self.padding:
            self.input_dim -= 1

        self.alpha = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.DoubleTensor(1, self.rank)))

        core_list = []
        for i in range(0, self.length):
            core = torch.nn.init.xavier_normal_(torch.DoubleTensor(self.rank, self.input_dim, self.rank))
            # for j in range(core.shape[1]):
            #     core[:, j, :] += torch.eye(self.rank)
            if self.padding:
                core[:, -1, :] =  torch.eye(self.rank)
            core_list.append(torch.nn.Parameter(core))

        self.omega =  torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.DoubleTensor(self.rank, 1)))

        self.core_list = core_list
        self._get_params()

    def _get_params(self):
        self.params = nn.ParameterList([])
        for i in range(0, self.length):
            self.params.append(self.core_list[i])
        self.params.append(self.alpha)
        self.params.append(self.omega)
        return

    def traver_tt(self, x):
        merged = torch.einsum('ij, jkl -> ikl', self.alpha, self.core_list[0])
        merged = torch.einsum('ijk, nj->nik', merged, x[:, 0, :])
        for k in range(1,len(self.core_list)):
            merged = torch.einsum('nik, kjl, nj -> nil', merged, self.core_list[k], x[:, k, :])
        return torch.einsum('nil, lk -> nik', merged, self.omega).squeeze()

    def get_tt(self):
        cores = []
        for core in self.core_list:
            cores.append(core.detach().numpy())
        alpha = self.alpha.detach().numpy()
        omega = self.omega.detach().numpy()
        cores[0] = np.einsum('ij, jkl -> ikl', alpha, cores[0])
        cores[-1] = np.einsum('ijk, kl -> ijl', cores[-1], omega)
        return cores

    def get_2norm(self):
        tmp = torch.einsum('ij, jkl -> ikl', self.alpha, self.core_list[0])
        tmp = torch.einsum('ikl, nkm->inlm', tmp, tmp)
        for i in range(1, len(self.core_list)):
            tmpi =  torch.einsum('ikl, nkm -> inlm', self.core_list[i], self.core_list[i])
            tmp = torch.einsum('inlm, lmkq -> inkq', tmp, tmpi)

        tmp = torch.einsum('inkq, kl, qp -> inlp', tmp, self.omega, self.omega).squeeze()
        return tmp

    def get_size_full_tensor(self):
        return self.input_dim ** self.length

    def loss_func(self, pred, y):
        H_norm = self.get_2norm()
        mse = torch.sum((pred - y) **2)/len(y)
        # loss = 0.5*H_norm - torch.sum(torch.mul(pred, y))
        loss = mse + 1./(self.get_size_full_tensor() - len(y))*(H_norm - torch.sum(pred ** 2))
        return loss

    def forward(self, x):
        if isinstance(x, list):
            new_x = []
            for i in range(len(x)):
                tmp = x[i].numpy()
                new_x.append(tmp)
            x = torch.DoubleTensor(np.asarray(new_x))
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).double()
        n = x.shape[0]
        l = x.shape[1]
        d = x.shape[2]
        x = x.reshape(n*l, d)

        x = x.reshape(n, l, d)
        x = self.traver_tt(x)
        return x

    def learning(self, x, y, scheduler_param, lr = 0.001, epochs = 1000, verbose = True, weight_decay = 1e-5):
        option = {
            # 'gamma':0.1
            'factor': 0.1,
            'patience': 10,
            'threshold':0.00001
        }
        scheduler_params = {**option, **scheduler_param}

        train_option = {
            'epochs': epochs,
            'verbose': verbose
        }

        train_data = Dataset(data=[x, y])
        test_data = Dataset(data=[x, y])
        generator_params = {'batch_size': 256,
                            'shuffle': True,
                            'num_workers': 1}
        train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
        test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

        optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True, weight_decay = weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        train_lambda = lambda model: train(model, 'cpu', train_loader, optimizer,
                                           torch.nn.MSELoss())
        validate_lambda = lambda model: validate(model, 'cpu', test_loader, torch.nn.MSELoss())
        train_loss_vec, validate_loss_vec = train_validate(self, train_lambda, validate_lambda, scheduler, train_option)
        return train_loss_vec, validate_loss_vec

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
        # norm *= torch.norm(self.omega, p = 'fro')




def get_hankel(train_file, l, pt, option, lr = 0.1, epochs = 1000,  lrows = 1, lcolumns = 1, exact_length = False, version ='classic'):
    if version == 'classic':
        x, y = get_xy_from_splearn(train_file, l, nbL=pt.nbL, rank=2, lrows=lrows, lcolumns=lcolumns, exact_length=exact_length)
    elif version == 'factor':
        x, y = get_xy_from_splearn_factor(train_file, l, nbL=pt.nbL, rank=2, lrows=lrows, lcolumns=lcolumns,
                                   exact_length=exact_length)
    # option['padding'] = not exact_length
    mdl = SGD_Hankel_v2(**option)
    print(mdl.core_list[0].shape)
    print(x.shape)
    train_loss, vali_loss = mdl.learning(x, y, scheduler_params, lr=lr, epochs=epochs, verbose=True, weight_decay=0.)
    H_l = mdl.get_tt()
    with open('hankel'+str(l)+'training loss', 'wb') as f:
        pickle.dump([train_loss, vali_loss], f)
    return H_l


if __name__ == '__main__':
    l = 3
    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    epochs = 1000
    load = False
    version = 'factor'
    pt = load_data_sample(train_file)
    exact_length = True
    option= {
        'length': l,
        'rank': 12,
        'input_dim': pt.nbL + 1,
        'out_dim': 1,
        'init_std': 0.01,
        'padding': True,
        'device': 'cpu'
    }
    if exact_length:
        option['padding'] = False
    # print(option)
    scheduler_params = {
        # 'gamma': 0.0001
        'factor': 0.1,
        'patience': 50,
        'threshold': 0.00001
    }
    # print(load)
    if not load:
        H_l = get_hankel(train_file, l, pt, option, lr=0.1, epochs=400, lrows=100, lcolumns=100, exact_length=exact_length, version=version)
        with open('sgdv2_hl'+version, 'wb') as f:
            pickle.dump(H_l, f)
        option['length'] = 2*l
        scheduler_params['factor'] = 0.5
        scheduler_params['patience'] = 100
        H_2l = get_hankel(train_file, 2*l, pt, option, lr=0.01, epochs=10*epochs, lrows=100, lcolumns=100, exact_length=exact_length, version=version)
        with open('sgdv2_h2l'+version, 'wb') as f:
            pickle.dump(H_2l, f)
        option['length'] = 2 * l + 1
        H_2l1 = get_hankel(train_file, 2*l+1, pt, option, lr=0.01, epochs=10*epochs, lrows=100, lcolumns=100, exact_length=exact_length,
                         version=version)
        with open('sgdv2_h2l1'+version, 'wb') as f:
            pickle.dump(H_2l1, f)
    else:
        with open('sgdv2_hl'+version, 'rb') as f:
            H_l = pickle.load(f)
        with open('sgdv2_h2l'+version, 'rb') as f:
            H_2l = pickle.load(f)
        with open('sgdv2_h2l1'+version, 'rb') as f:
            H_2l1 = pickle.load(f)

    print(tt_to_tensor(H_l).shape)

    model = spectral_learning_matrix(tt_to_tensor(H_l), tt_to_tensor(H_2l), tt_to_tensor(H_2l1), rank=12)
    model.factor_to_classic()
    error = test_spice(model, test_file, solution_file, sp=False)
    print(error)