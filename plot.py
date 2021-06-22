from matplotlib import pyplot as plt
import pickle
import numpy as np
import os
import random
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_files(Files):
    count = 0
    for file in Files:
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                error = np.asarray(pickle.load(f))[2:]
            train_MSE= error[:, 0][:][0]
            train_MAPE = error[:, 0][:][1]
            # print(error[:, 0][])
            # print(error[:, 1])
            # print(error[:, 2])
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(train_MSE,  '--', label = file + '_training_MSE', color = color)
            plt.plot(error[:, 1], label=file + '_testing_MSE', color=color)


            count += 1
        # print(file)

    plt.legend(loc  = 'upper right')
    plt.xlabel('num_epochs')
    plt.ylabel('MSE')
    plt.yscale('log')
    plt.show()

    for file in Files:
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                error = np.asarray(pickle.load(f))[2:]
            train_MSE= error[:, 0][:][0]
            train_MAPE = error[:, 0][:][1]
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(train_MAPE, '--', label=file + '_training_MAPE', color=color)
            plt.plot(error[:, 2], label=file + '_testing_MAPE', color=color)
    plt.legend(loc='upper right')
    plt.xlabel('num_epochs')
    plt.ylabel('MAPE')
    plt.yscale('log')
    plt.show()

def get_best_results(Files):
    for file in Files:
        if os.path.isfile(file):
            print(file)
            with open(file, 'rb') as f:
                error = np.asarray(pickle.load(f))[2:]
            train_MSE= error[:, 0][:][0]
            train_MAPE = error[:, 0][:][1]
            test_MSE = error[:, 1]
            test_MAPE = error[:, 2]
            print(min(test_MSE), min(test_MAPE))

def get_results(Files):
    print(Files)
    training_error = []
    for file in Files:
        print(file)
        if os.path.isfile(file):
            with open(file, 'rb') as f:
                error = np.asarray(pickle.load(f))[2:]
                for i in range(len(error)):
                    training_error.append(error[i][1])
                plt.plot(training_error, label='N_300000_l_50_d_2_r_4' + '_training_MAPE')
                plt.plot(error[:, -1], label='N_300000_l_50_d_2_r_4' + '_testing_MAPE')
    plt.legend(loc='upper right')
    plt.xlabel('num_epochs')
    plt.ylabel('MAPE')
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    Ns = [300000]
    ls = [40]
    ds = [2]
    rs = [4]

    param_list = []
    for n in Ns:
        for l in ls:
            for d in ds:
                for r in rs:
                    param_list.append([n, l, d, r])
    file_names = []
    for param in param_list:
        # print(paramd= 5)
        n = param[0]
        l = param[1]
        d = param[2]
        r = param[3]
        # file_name = 'als_' + str(n) + '_' + str(l) + '_' + str(d) + '_' + str(r) + '.error'
        file_name = 'als_' + str(n) + '_' + str(l) + '_' + str(d) + '_' + str(r) + '.error7'
        file_names.append(file_name)
        # for i in range(1, 8):
        #     file_names.append(file_name + str(i))
    get_results(file_names)

    # plot_files(file_names)
