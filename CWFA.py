import numpy as np
# import tensorly as tl
from torch import nn
import torch
from sklearn.linear_model import LinearRegression
from torch import optim
import torch.nn.functional as F
from gradient_descent import train, validate, fit
from Dataset import Dataset


class CWFA(nn.Module):

    def __init__(self, **option):
        super(CWFA, self).__init__()
        option_default = {
            'A': None,
            'alpha': None,
            'Omega': None,
            'example_WFA': False,
            'rank': 5,
            'device': 'cpu',
            'input_dim': 3,
            'init_std': 0.1,
            'out_dim': 1
        }
        option = {**option_default, **option}
        self.device = option['device']
        self.output_dim = option['out_dim']
        self.rank = option['rank']
        self.input_dim = option['input_dim']
        A = option['A']
        alpha = option['alpha']
        Omega = option['Omega']

        if option['example_WFA']:
            self.A = torch.nn.parameter.Parameter(
                torch.tensor(torch.normal(0.1, option['init_std'], [self.rank, self.input_dim, self.rank]),
                             requires_grad=True)).to(self.device)
            self.alpha = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0.1, option['init_std'], [self.rank]),
                                                                   requires_grad=True)).to(self.device)
            self.Omega = torch.nn.parameter.Parameter(
                torch.tensor(torch.normal(0.1, option['init_std'], [self.rank, self.output_dim]),
                             requires_grad=True)).to(self.device)
        else:
            if A is not None:
                if isinstance(A, np.ndarray):
                    A = torch.from_numpy(A)
                self.A = A
            else:
                self.A = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, option['init_std'], [self.rank, self.input_dim, self.rank]),
                    requires_grad=True)).to(self.device)

            if alpha is not None:
                if isinstance(alpha, np.ndarray):
                    alpha = torch.from_numpy(alpha)
                self.alpha = alpha
            else:
                self.alpha = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, option['init_std'], [self.rank]),
                                                                  requires_grad=True)).to(self.device)

            if Omega is not None:
                if isinstance(Omega, np.ndarray):
                    Omega = torch.from_numpy(Omega)
                self.Omega = Omega
            else:
                self.Omega = torch.nn.parameter.Parameter(
                    torch.tensor(torch.normal(0, option['init_std'], [self.rank, self.output_dim]),
                                 requires_grad=True)).to(self.device)

        self.alpha.reshape([self.rank, ])

    def forward(self, x):
        '''
        :param x: of shape N, L, D
        :return:
        '''
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).double()
        tmp = torch.einsum('i, ijk, nj ->nk', self.alpha, self.A, x[:, 0, :])
        for i in range(1, x.shape[1]):
            # print(tmp.shape, self.A.shape, x[:, i, :].shape)
            tmp = torch.einsum('ni, ijk, nj ->nk', tmp, self.A, x[:, i, :])
        tmp = torch.einsum('ni, im -> nm', tmp, self.Omega)

        return tmp.squeeze()

    def _get_params(self):
        self.params = nn.ParameterList([])
        self.params.append(self.alpha)
        self.params.append(self.Omega)
        self.params.append(self.A)
        return

def obtain_outerproduct_index(l = 5, d = 3):
    indexes = np.random.randint(0, d, size=l)
    empty_index = np.random.randint(0, l, size = 1)[0]
    indexes[empty_index] = -1
    return indexes

def obtain_input_basedon_outerproduct_index(x, indexes):
    tmp = None
    for i in range(x.shape[1]):
        if indexes[i] == -1:
            tmp = x[:, i, :]
            break
    if tmp is None:
        print('cant find -1 in index list')
        exit()

    for i in range(x.shape[1]):

        current_dim = indexes[i]
        if indexes[i] == -1:
            continue
        else:
            tmp = np.einsum('ni, n->ni', tmp, x[:, i, current_dim])
    return tmp

def get_combination_data(x, y, l, d, num_combinations = 10):
    indexes = obtain_outerproduct_index(l, d)
    new_x = obtain_input_basedon_outerproduct_index(x, indexes)
    new_y = y
    for i in range(num_combinations):
        indexes = obtain_outerproduct_index(l, d)
        tmp_x = obtain_input_basedon_outerproduct_index(x, indexes)

        new_x = np.concatenate((new_x, tmp_x), axis=0)
        new_y = np.concatenate((new_y, y), axis=0)
    return new_x, new_y

def get_linear_regressor(x, y):
    reg = LinearRegression(fit_intercept = False).fit(x, y)
    return reg

def exam_regressor(x, y, reg):
    pred = reg.predict(x)
    # print('R2 score: ', reg.score(x, y))
    mse = np.mean((y.reshape(-1, 1) - pred.reshape(-1, 1)) ** 2)
    # print('MSE: ', mse)
    # print('First 5 predictions: ', y[:5], pred[:5])
    return mse

def learn_cwfa(cwfa, train_loader, validate_loader, **fit_option):
    option_default = {
        'train_loss': F.mse_loss,
        'vali_loss': F.mse_loss,
        'lr': 0.001
    }
    fit_option = {**option_default, **fit_option}
    optimizer = optim.Adam(cwfa.parameters(), lr=fit_option['lr'], amsgrad=True)
    train_lambda = lambda model: train(model, cwfa.device, train_loader, optimizer, fit_option['train_loss'])
    validate_lambda = lambda model: validate(model, cwfa.device, validate_loader, fit_option['vali_loss'])
    fit_option['optimizer'] = optimizer
    cwfa = fit(cwfa, train_lambda, validate_lambda, **fit_option)
    return cwfa

if __name__ == '__main__':
    option_default = {
        'A': None,
        'alpha': None,
        'Omega': None,
        'example_WFA': True,
        'rank': 5,
        'device': 'cpu',
        'input_dim': 3,
        'init_std': 0.1,
        'out_dim': 1
    }
    n = 100
    l = 5
    d = option_default['input_dim']
    cwfa = CWFA(**option_default)
    x = np.random.rand(n, l, d)
    y = cwfa(x).detach().numpy()
    test_x = np.random.rand(n, l, d)
    test_y = cwfa(x).detach().numpy()
    x[:, :, 0] = 1
    x, y = get_combination_data(x, y, l, d, num_combinations=10)
    test_x, test_y = get_combination_data(test_x, test_y, l, d, num_combinations=1)
    reg = get_linear_regressor(x, y)
    print('Training Error: ', exam_regressor(x, y, reg))
    print('Testing Error: ', exam_regressor(test_x, test_y, reg))
    W = reg.coef_
    U, D, V = np.linalg.svd(W)
    A_m = np.diag(D) @ V
    print(A_m.shapeg)

