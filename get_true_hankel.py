from splearn.datasets.base import load_data_sample
from utils import *
import numpy as np
from ALS_CWFA import spectral_learning_matrix, tt_to_tensor, test_spice
import pickle


def matching_hankel(H):
    dim = len(H.shape)
    nbl = H.shape[0]
    tmp = []
    for i in range(dim):
        tmp2 = []
        for j in range(dim):
            if i == j:
                tmp2.append(slice(nbl-1, nbl))
            else:
                tmp2.append(slice(None))
        tmp.append(tmp2)
    # print(H.shape)
    # print(tmp)
    lambda_layer = []
    for slicing in tmp:
        slicing = tuple(slicing)
        # print(H[slicing].shape)
        lambda_layer.append(H[slicing].squeeze())

    lambda_layer = np.asarray(lambda_layer)
    # print(lambda_layer.shape)
    max_lambda = np.max(lambda_layer, axis = 0)
    # print(max_lambda)

    for slicing in tmp:
        slicing = tuple(slicing)
        H[slicing] = max_lambda.reshape(H[slicing].shape)
    return H



def get_true_hankel(x, y, l, nbL):

    hankel_shape = []
    for i in range(l):
        hankel_shape.append(nbL)
    H = np.zeros(hankel_shape)

    for i in range(len(x)):
        loc = x[i][0]
        for j in range(1, len(x[i])):
            loc = np.kron(loc, x[i][j])
        loc = loc.reshape(hankel_shape)
        loc = np.asarray(loc, dtype=bool)
        # print(loc)
        # print(np.bool(loc))
        H[loc] = y[i]
    # H = matching_hankel(H)
    # H = remove_lambda(H)
    return H

def remove_lambda(H):
    tmp = []
    for i in range(len(H.shape)):
        tmp.append(slice(-1))

    return H[tuple(tmp)]

if __name__ == '__main__':
    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    l = 3
    pt = load_data_sample(train_file)
    exact_length = True
    version = 'factor'
    if version == 'factor':
        x, y = get_xy_from_splearn_factor(train_file, l, pt.nbL, rank=2, lrows=100, lcolumns=100, exact_length=exact_length, mnne_factor= 0)
        if exact_length:
            H_l = get_true_hankel(x, y, l, pt.nbL)
        else:
            H_l = get_true_hankel(x, y, l, pt.nbL+1)
        print(H_l)
        print(np.count_nonzero(H_l)/H_l.size, H_l.size)

        x, y = get_xy_from_splearn_factor(train_file, 2*l, pt.nbL, rank=2, lrows=100, lcolumns=100, exact_length=exact_length, mnne_factor= 0)
        if exact_length:
            H_2l = get_true_hankel(x, y, 2*l, pt.nbL)
        else:
            H_2l = get_true_hankel(x, y, 2*l, pt.nbL+1)
        # H = H_2l.reshape(5**l, 4**l)

        print(np.count_nonzero(H_2l) / H_2l.size, H_2l.size)


        x, y = get_xy_from_splearn_factor(train_file, 2*l+1, pt.nbL, rank=2, lrows=100, lcolumns=100, exact_length=exact_length, mnne_factor= 0)
        if exact_length:
            H_2l1 = get_true_hankel(x, y, 2 * l+1, pt.nbL)
        else:
            H_2l1 = get_true_hankel(x, y, 2 * l+1, pt.nbL+1)
        print(np.count_nonzero(H_2l1) / H_2l1.size, H_2l1.size)

        model = spectral_learning_matrix(H_l, H_2l, H_2l1, rank=12)
        model.factor_to_classic()
    else:
        x, y = get_xy_from_splearn(train_file, l, pt.nbL, rank=2, lrows=100, lcolumns=100,
                                          exact_length=exact_length, mnne_factor=0)

        if exact_length:
            H_l = get_true_hankel(x, y, l, pt.nbL)
        else:
            H_l = get_true_hankel(x, y, l, pt.nbL+ 1)
        print(H_l)
        print(np.count_nonzero(H_l) / H_l.size, H_l.size)

        x, y = get_xy_from_splearn(train_file, 2 * l, pt.nbL, rank=2, lrows=100, lcolumns=100,
                                          exact_length=exact_length, mnne_factor=0)
        if exact_length:
            H_2l = get_true_hankel(x, y, 2 * l, pt.nbL)
        else:
            H_2l = get_true_hankel(x, y, 2 * l, pt.nbL + 1)
        # H = H_2l.reshape(5**l, 4**l)

        print(np.count_nonzero(H_2l) / H_2l.size, H_2l.size)

        x, y = get_xy_from_splearn(train_file, 2 * l + 1, pt.nbL, rank=2, lrows=100, lcolumns=100,
                                          exact_length=exact_length, mnne_factor=0)
        if exact_length:
            H_2l1 = get_true_hankel(x, y, 2 * l + 1, pt.nbL)
        else:
            H_2l1 = get_true_hankel(x, y, 2 * l + 1, pt.nbL + 1)
        print(np.count_nonzero(H_2l1) / H_2l1.size, H_2l1.size)

        model = spectral_learning_matrix(H_l, H_2l, H_2l1, rank=12)
    error = test_spice(model, test_file, solution_file, sp=exact_length)

    with open('true_hankel'+version, 'wb') as f:
        pickle.dump([H_l, H_2l, H_2l1], f)


    print(error)


