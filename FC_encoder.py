import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self, **option):
        super(Encoder, self).__init__()
        option_default = {
            'input_dim': 5,
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        }
        option = {**option_default, **option}
        self.encoder = []
        self.out_dim = option['out_dim']
        self.input_dim = option['input_dim']
        option['hidden_units'].insert(0, option['input_dim'])
        option['hidden_units'].append(option['out_dim'])
        self.num_neurons = option['hidden_units']
        for i in range(len(self.num_neurons) - 1):
            self.encoder.append(nn.Linear(self.num_neurons[i], self.num_neurons [i+1]))
        self.inner_activation = option['inner_activation']
        self.final_activation = option['final_activation']

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            # if i < len(self.encoder) - 1:
            #     x = self.inner_activation(x)
            # else:
            #     x = self.final_activation(x)
        return x

    def _get_params(self):
        self.params = nn.ParameterList([])
        for i in range(len(self.encoder)):
            self.params.append(self.encoder[i].weight)
            self.params.append(self.encoder[i].bias)
        return