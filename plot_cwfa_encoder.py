from matplotlib import pyplot as plt
import numpy as np
import pickle
import os

def get_results(File_names, als_epochs = 10, nn_epochs = 100, all_epochs = 10):
    train_ALS = []
    validate_ALS = []
    train_nn = []
    validate_nn = []

    for File in File_names:
        train = []
        validate = []
        if os.path.isfile(File):
            with open(File, 'rb') as f:
                error = pickle.load(f)
                ALS_error = error['ALS_error']
                nn_error = error['neuralnets_error']

                # print(np.asarray(nn_error).shape)
                count_ALS = 1
                count = 1
                curr = 'ALS'
                for i in range(all_epochs):
                    for j in range(als_epochs):
                        # print(ALS_error[i][j][1])
                        train.append(ALS_error[i][j][0][0].numpy())
                        validate.append(ALS_error[i][j][1])
                    for k in range(nn_epochs - 1):
                        # print(len(nn_error[i]))
                        # print(nn_error[i][1])
                        # print(nn_error[i+1][k])
                        train_index = i*2
                        if isinstance(nn_error[train_index][k], float):
                            train.append(nn_error[train_index][k])
                        else:
                            train.append(nn_error[train_index][k].detach().numpy())
                        validate.append(nn_error[train_index + 1][k])

                #
                # for i in range(30*1000 - 1):
                #     if (int(count /100))%2 ==0:
                #         index = int(count /100)
                #         print(i, i-index*100, index)
                #         print(len(ALS_error), len(ALS_error[0]), len(ALS_error[0][0]))
                #         train.append(ALS_error[index][i - index*100][1])
                #         validate.append(ALS_error[index][i- index*100][-1])
                #         count_ALS += 1
                #     else:
                #         index = int(count /100)
                #         train.append(nn_error[index][i - index*100+1])
                #         validate.append(nn_error[index+1][i - index * 100+1])
                #     count += 1
                train = np.asarray(train).reshape(-1,)

                plt.plot(train, label = 'training')
                plt.plot(validate, label = 'validation')
                plt.ylabel('MSE')
                plt.xlabel('Epochs')
                plt.yscale('log')
                plt.title(File)
                plt.legend()
                plt.show()


if __name__ == '__main__':
    Ns = [10000, 30000, 50000]
    ls = [10]
    ds = [3]
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
        file_name = 'cwfa_encoder' + str(n) + '_' + str(l) + '_' + str(d) + '_' + str(r) + '.error_decay5'
        file_names.append(file_name)
        # for i in range(1, 8):
        #     file_names.append(file_name + str(i))
    get_results(file_names)