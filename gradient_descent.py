import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import optim
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def MAPE(pred, y):
    err = pred - y
    err = np.divide(err, y)
    return np.mean(np.abs(err))

def train(model, device, train_loader, optimizer, loss_function = F.mse_loss):
    error = []

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device),y.to(device)
        # y_mean = torch.mean(y ** 2)
        optimizer.zero_grad()
        output = model(x).to(device)


        # mape = MAPE(output.detach().numpy(), y.numpy())

        # param.grad[:, 1:10, :, :] = 0

        # regularization_loss = 0
        # for param in model.parameters():
        #     regularization_loss += torch.sum(abs(param))
        # loss = loss_function(output, y)  + 1.*(model.get_2norm() - torch.sum(output**2))
        loss = model.loss_func(output, y)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0)
        # total_norm = []
        # for param in model.parameters():
        #     # param_norm = param.grad.detach().data.norm(2)
        #     # total_norm.append(param_norm **0.5)
        #     # total_norm += param_norm.item() ** 2
        #
        #
        #     if len(param.data.shape) == 3:
        #         if param.data.shape[0] == param.data.shape[-1]:
        #             param.grad.data[:, -1, :] = 0
                    # print(param.data[:, -1, :])
        # total_norm = total_norm ** 0.5
        # print(total_norm)
        optimizer.step()
        error.append(loss)
    # print(optimizer.param_groups[0]['lr'])
        # model.normalize()
        # print(model.core_list[0][:, -1, :])
    # print(output[:5])
    # print(y[:5])
    return sum(error) / len(error)


def validate(model, device, test_loader, loss_function = F.mse_loss):
    # test_loss = 0
    all_losses = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device),y.to(device)
            # y_mean = torch.mean(y ** 2)
            output = model(x).to(device)
            test_loss = loss_function(output, y).item()
            #mape = MAPE(output.detach().numpy(), y.numpy())
            all_losses.append(test_loss)
    # mape /= len(x)

    return sum(all_losses) / len(all_losses)

def train_validate (model, train_lambda, validate_lambda, scheduler, option):
    option_default = {
        'verbose': True,
        'epochs': 1000
    }
    option = {**option_default, **option}
    train_loss_vec = []
    validate_loss_vec = []
    for epoch in range(1, option['epochs'] + 1):
        train_loss = train_lambda(model)
        validate_loss = validate_lambda(model)
        if option['verbose']:
            print('Epoch: '+str(epoch)+'Train Error: {:.10f} Validate Error: {:.10f}'.format(train_loss, validate_loss))
        scheduler.step(validate_loss)

        # scheduler.step()
        train_loss_vec.append(train_loss)

        validate_loss_vec.append(validate_loss)

    return train_loss_vec, validate_loss_vec

def fit(model, train_lambda, validate_lambda, **option):
    option_default = {
        'step_size': 500,
        'gamma': 0.1,
        'epochs': 1000,
        'verbose': True,
        'optimizer': optim.Adam(model.parameters(), lr=0.001, amsgrad=True),
    }
    option = {**option_default, **option}

    scheduler_params = {
        'step_size': option['step_size'],
        'gamma': option['gamma']
    }
    train_option = {
        'epochs': option['epochs'],
        'verbose': option['verbose']
    }
    scheduler = StepLR(option['optimizer'], **scheduler_params)
    train_loss_vec, validate_loss_vec = model.fit(train_lambda, validate_lambda, scheduler, **train_option)
    return model, train_loss_vec, validate_loss_vec