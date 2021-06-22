import numpy as np
import torch
from torch import nn
from gradient_descent import *
from Dataset import Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from TT_learning import TT_spectral_learning
from CWFA import CWFA
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

class SGD_Hankel(nn.Module):
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

        core_list = [torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.DoubleTensor(1, self.input_dim, self.rank)))]
        for i in range(1, self.length - 1):
            core = torch.nn.init.xavier_normal_(torch.DoubleTensor(self.rank, self.input_dim, self.rank))
            if self.padding:
                core[:, -1, :] =  torch.eye(self.rank)
            core_list.append(torch.nn.Parameter(core))

        core = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.DoubleTensor(self.rank, self.input_dim, self.out_dim)))
        core_list.append(core)
        self.core_list = core_list
        self._get_params()

    def _get_params(self):
        self.params = nn.ParameterList([])
        self.params.append(self.core_list[0])
        for i in range(1, self.length-1):
            self.params.append(self.core_list[i])
        self.params.append(self.core_list[-1])
        return

    def contract_in_range(self, x, start_index, end_index):
        # print( self.core_list[start_index].shape,  x[:, start_index, :].shape)
        merged = torch.einsum('ijk, nj->nik', self.core_list[start_index], x[:, start_index, :])
        # print('0', merged[0, 0, 0])
        for k in range(start_index+1,end_index):
            # print(self.core_list[k].shape)
            # print(x.shape, k)
            merged = torch.einsum('nik,kjl, nj -> nil', merged, self.core_list[k], x[:, k, :])
            # print(k ,merged[0, 0, 0])
        return merged

    def get_weight(self):
        cores = []
        for core in self.core_list:
            cores.append(core.detach().numpy())
        return cores

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
        x = self.contract_in_range(x, 0, self.length).squeeze()
        return x



def learning(x, y, h, scheduler_param, lr = 0.001, epochs = 1000, verbose = True, weight_decay = 1e-5):
    option = {
        # 'gamma':0.1
        'factor': 0.1,
        'patience': 10,
        'threshold':0.00001
    }
    scheduler_params = {**option, **scheduler_param}


    # scheduler_params = {
    #     'step_size': 200,
    #     'gamma': 0.1
    # }

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

    optimizer = optim.Adam(h.parameters(), lr=lr, amsgrad=True, weight_decay = weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    train_lambda = lambda model: train(model, 'cpu', train_loader, optimizer,
                                       torch.nn.MSELoss())
    validate_lambda = lambda model: validate(model, 'cpu', test_loader, torch.nn.MSELoss())
    train_loss_vec, validate_loss_vec = train_validate(h, train_lambda, validate_lambda, scheduler, train_option)
    return h



if __name__ == '__main__':
    import pickle
    load = False
    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    exact_length = False
    epochs = 1000
    version = 'factor'
    l = 3
    pt = load_data_sample(train_file)


    if exact_length:
        option = {
            'length': l,
            'rank': 12,
            'input_dim': pt.nbL,
            'out_dim': 1,
            'padding': True
        }
    else:
        option = {
            'length': l,
            'rank': 12,
            'input_dim': pt.nbL + 1,
            'out_dim': 1,
            'padding': True
        }



    if load:
        with open('H2l', 'rb') as f:
            H_2l = pickle.load(f)
        with open('H2l1', 'rb') as f:
            H_2l1 = pickle.load(f)
        with open('Hl', 'rb') as f:
            H_l = pickle.load(f)
    else:
        scheduler_params = {
            # 'gamma': 0.0001
            'factor': 0.1,
            'patience': 10,
            'threshold': 0.00001
        }
        if version == 'classic':
            x, y = get_xy_from_splearn(train_file, l, pt.nbL, rank=2, lrows=1, lcolumns=1, exact_length=exact_length)
        else:
            x, y = get_xy_from_splearn_factor(train_file, l, pt.nbL, rank=2, lrows=100, lcolumns=100, exact_length=exact_length)
        h = SGD_Hankel(**option)
        H_l = learning(x, y, h, scheduler_param=scheduler_params, lr=0.1, epochs=int(2*epochs), weight_decay =0)
        H_l = H_l.get_weight()
        # H_l = remove_lambda(H_l)
        tmp  = tt_to_tensor(H_l)
        print(tmp)
        with open('Hl', 'wb') as f:
            pickle.dump(H_l, f)



        scheduler_params = {
            # 'gamma': 0.0001
            'factor': 0.1,
            'patience': 100,
            'threshold': 0.00001
        }
        if version == 'classic':
            x, y = get_xy_from_splearn(train_file, 2*l, pt.nbL, rank=2, lrows=1, lcolumns=1, exact_length=exact_length)
        else:
            x, y = get_xy_from_splearn_factor(train_file, 2*l, pt.nbL, rank=2, lrows=100, lcolumns=100,
                                       exact_length=exact_length)
        option['length'] = 2 * l
        print(l, option['length'])
        h = SGD_Hankel(**option)
        H_2l = learning(x, y, h, scheduler_param=scheduler_params, lr=0.1, epochs=2*epochs, weight_decay = 0)
        H_2l = H_2l.get_weight()
        # H_2l = remove_lambda(H_2l)
        with open('H2l', 'wb') as f:
            pickle.dump(H_2l, f)

        scheduler_params = {
            # 'gamma': 0.001
            'factor': 0.1,
            'patience': 50,
            'threshold': 0.00001
        }
        if version == 'classic':
            x, y = get_xy_from_splearn(train_file, 2*l+1, pt.nbL, rank=2, lrows=1, lcolumns=1, exact_length=exact_length)
        else:
            x, y = get_xy_from_splearn_factor(train_file, 2*l+1, pt.nbL, rank=2, lrows=100, lcolumns=100,
                                       exact_length=exact_length)
        option['length'] = 2 * l + 1
        h = SGD_Hankel(**option)
        H_2l1 = learning(x, y, h, scheduler_param=scheduler_params, lr=0.1
                         , epochs=2*epochs, weight_decay = 0)
        H_2l1 = H_2l1.get_weight()
        # H_2l1 = remove_lambda(H_2l1)


        with open('H2l1', 'wb') as f:
            pickle.dump(H_2l1, f)



    model = spectral_learning_matrix(tt_to_tensor(H_l), tt_to_tensor(H_2l), tt_to_tensor(H_2l1), rank=12)
    model.factor_to_classic()
    error = test_spice(model, test_file, solution_file, sp= True)
    print(error)







