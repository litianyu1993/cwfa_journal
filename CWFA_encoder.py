from CWFA import CWFA
from ALS_CWFA import ALS_CWFA
import pickle
import numpy as np
import torch
from Dataset import Dataset
from torch import nn
from gradient_descent import *

class ALS_CWFA_Encoder(nn.Module):

    def __init__(self, hankel, target = False, **option):
        super().__init__()
        self.hankel = hankel
        option_default = {
            'input_dim': 5,
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False),
        }
        option = {**option_default, **option}
        self.encoder = []
        self.out_dim = option['out_dim']
        self.input_dim = option['input_dim']
        option['hidden_units'].insert(0, option['input_dim'])
        option['hidden_units'].append(option['out_dim'])
        self.num_neurons = option['hidden_units']
        for i in range(len(self.num_neurons) - 1):
            self.encoder.append(nn.Linear(self.num_neurons[i], self.num_neurons[i + 1]))
            if target:
                torch.nn.init.normal_(self.encoder[-1].weight, std = 0.5)

        self.inner_activation = option['inner_activation']
        self.final_activation = option['final_activation']
        self._get_params()

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
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.encoder) - 1:
                x = self.inner_activation(x)
            # else:
            #     x = self.final_activation(x)
        x = x.reshape(n, l, d)
        x = self.hankel.contract_in_range(x, 0, self.hankel.length).squeeze()
        return x

    def get_ALS_input(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).double()
        n = x.shape[0]
        l = x.shape[1]
        d = x.shape[2]
        x = x.reshape(n * l, d)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.encoder) - 1:
                x = self.inner_activation(x)
            # else:
            #     x = self.final_activation(x)
        x = x.reshape(n, l, d)
        return x

    def fit(self, train_lambda, validate_lambda, scheduler, **option):

        default_option = {
            'verbose': True,
            'epochs': 1000
        }
        option = {**default_option, **option}
        train_loss_vec, validate_loss_vec = train_validate(self, train_lambda, validate_lambda, scheduler, option)
        return train_loss_vec, validate_loss_vec

    def _get_params(self):
        self.params = nn.ParameterList([])
        for i in range(len(self.encoder)):
            self.params.append(self.encoder[i].weight)
            self.params.append(self.encoder[i].bias)
        return

def learning_encoder(cwfa_encoder, train_x, train_y, test_x, test_y, **fit_option):
    # if not torch.is_tensor(train_x):
    #     train_x = torch.from_numpy(train_x).double()
    # if not torch.is_tensor(train_y):
    #     train_y = torch.from_numpy(train_y).double()
    # if not torch.is_tensor(test_x):
    #     test_x = torch.from_numpy(test_x).double()
    # if not torch.is_tensor(test_y):
    #     test_y = torch.from_numpy(test_y).double()

    train_data = Dataset(data=[train_x, train_y])
    test_data = Dataset(data=[test_x, test_y])
    generator_params = {'batch_size': fit_option['batch_size'],
                        'shuffle': fit_option['shuffle'],
                        'num_workers': 1}
    train_loader = torch.utils.data.DataLoader(train_data, **generator_params)
    test_loader = torch.utils.data.DataLoader(test_data, **generator_params)

    optimizer = optim.Adam(cwfa_encoder.parameters(), lr=fit_option['lr'], amsgrad=True)
    train_lambda = lambda model: train(model, 'cpu', train_loader, optimizer,
                                       torch.nn.MSELoss())
    validate_lambda = lambda model: validate(model, 'cpu', test_loader, torch.nn.MSELoss())
    fit_option['optimizer'] = optimizer
    cwfa_encoder, train_loss_vec, validate_loss_vec= fit(cwfa_encoder, train_lambda, validate_lambda, **fit_option)
    return cwfa_encoder, train_loss_vec, validate_loss_vec

def MAPE(pred, y):
    err = pred - y
    err = np.divide(err, y)
    # print(pred[:3], y[:3])
    return np.mean(np.abs(err))

if __name__ == '__main__':
    all_epochs = 10
    epochs = 100
    Ns = [10000, 30000, 50000]
    ls = [10]
    ds = [3]
    rs = [4]
    # Ns = [10000, 20000, 40000, 80000, 160000]
    # ls = [10, 20, 30]
    # ds = [5, 10]
    # rs = [4]
    param_list = []
    for n in Ns:
        for l in ls:
            for d in ds:
                for r in rs:
                    param_list.append([n, l, d, r])
    for param in param_list:
        n = param[0]
        l = param[1]
        d = param[2]
        r = param[3]
        encoder_option = {
            'input_dim': d,
            'hidden_units': [10,10],
            'out_dim': d,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        }

        option_default = {
            'A': None,
            'alpha': None,
            'Omega': None,
            'example_WFA': True,
            'rank': 3,
            'device': 'cpu',
            'input_dim': d,
            'init_std': 0.5,
            'out_dim': 1
        }
        fit_option= {
            'epochs': epochs,
            'verbose': True,
            'batch_size': 128,
            'shuffle': True,
            'num_workers':1,
            'lr': 0.0001,
            'step_size': 500,
            'gamma': 0.1
        }
        ALS_option = {
            'core_list': [],
            'length': l,
            'rank': r,
            'input_dim': d,
            'out_dim': 1,
            'init_std': 1,
            'cwfa': None
        }

        # d = option_default['input_dim']
        hankel_random = ALS_CWFA(**ALS_option)
        cwfa_encoder_random = ALS_CWFA_Encoder(hankel_random, target=False, **encoder_option)

        hankel_train = ALS_CWFA(**ALS_option)
        cwfa_encoder_train = ALS_CWFA_Encoder(hankel_train, target=False, **encoder_option)

        ALS_option['init_std'] = 1.

        hankel_target = ALS_CWFA(**ALS_option)
        cwfa_encoder_target = ALS_CWFA_Encoder(hankel_target, target = True, **encoder_option)

        x = np.ones((n, l, d))
        x[:, :, :-1] = np.random.normal(0, 1, (n, l, d-1))

        y = cwfa_encoder_target(x).detach().numpy()
        y_random = cwfa_encoder_random(x).detach().numpy()
        print(y[:3], y_random[:3])
        print('random baseline:', np.mean((y - y_random)**2))

        test_x = np.ones((10000, l, d))
        test_x[:, :, :-1] = np.random.normal(0, 1, (10000, l, d-1))
        test_y = cwfa_encoder_target(test_x).detach().numpy()

        error_all = []
        ALS_error = []
        neural_nets_error = []
        for j in range(0, all_epochs):
            als_learner = cwfa_encoder_train.hankel
            # print(als_learner.core_list[0][0])
            # print(hankel_target.core_list[0][0])
            als_x_train =  cwfa_encoder_train.get_ALS_input(x).detach().numpy()
            # print(als_x_train[:3])
            als_x_test = cwfa_encoder_train.get_ALS_input(test_x).detach().numpy()
            # print(als_x_train[:3])
            # print(y[:3])
            errors = als_learner.ALS_learning(als_x_train, y, epochs=10, test_x= als_x_test, test_y=test_y)
            ALS_error.append(errors)
            cwfa_encoder_train.hankel = als_learner
            cwfa_encoder_train, train_loss_vec, validate_loss_vec = learning_encoder(cwfa_encoder_train, x, y, test_x, test_y, **fit_option)
            neural_nets_error.append(train_loss_vec)
            neural_nets_error.append(validate_loss_vec)

            pred = cwfa_encoder_train(x).detach().numpy().reshape(-1,)
            mape = MAPE(pred, y.reshape(-1, ))
            mse = np.mean((pred - y.reshape(-1, ))**2)
            print('MAPE, ', mape)
            print('MSE, ', mse)
            error_all.append([mape, mse])
            print(y[:3], pred[:3])

            file_name = 'cwfa_encoder' + str(n) + '_' + str(l) + '_' + str(d) + '_' + str(r) + '.error_decay10'
            with open(file_name, 'wb') as f:
                pickle.dump({'error_all': error_all, 'ALS_error': ALS_error, 'neuralnets_error': neural_nets_error}, f)
            fit_option['lr'] /= 10