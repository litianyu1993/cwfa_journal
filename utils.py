import numpy as np
import copy
from splearn.datasets.base import load_data_sample
import splearn
import splearn.datasets.base as spb
def get_traj_plus_one_len(prefix, nbL):
    '''
    :param prefix: a prefix in list format
    :param nbL: number of letters
    :return:
    '''
    all_trajectories = []
    for j in range(nbL):
        tmp = copy.deepcopy(prefix)
        tmp.append(j)
        all_trajectories.append(tmp)
    return all_trajectories

def add_negative_examples_v2(x, y, max_length, nbL = 4, max_num_negative_examples = 10):
    all_trajs = get_traj_up_to_L([[]], nbL, max_length)
    idx = np.arange(len(all_trajs)).astype(int)
    idx = np.random.permutation(idx).astype(int)
    # print(idx)
    all_trajs = np.asarray(all_trajs)[idx.astype(int)]

    dict = {}
    for tmp in x:
        key = str(tmp)
        dict[key] = 1
    ori = copy.deepcopy(len(x))
    for traj in all_trajs:
        if len(traj) != max_length: continue
        if len(x) -  ori >= max_num_negative_examples:
            break
        key = str(traj)
        if key not in dict:
            dict[key] = 1
            x.append(traj)
            y = np.append(y, 0.)

    for traj in all_trajs:
        # if len(traj) != max_length: continue
        if len(x) -  ori >= max_num_negative_examples:
            break
        key = str(traj)
        if key not in dict:
            dict[key] = 1
            x.append(traj)
            y = np.append(y, 0.)
        if len(x) -  ori >= max_num_negative_examples:
            break
    return x, y


def pad_v2(x, y, symbol, max_length, mode = 'splearn'):
    new_x = np.asarray([])
    new_y = []
    if mode == 'splearn':
        for i in range(len(x)):
            tmpx = copy.deepcopy(x[i])
            tmp_padded = []
            for j in range(max_length - len(tmpx) +1):
                tmp = np.ones(max_length)*symbol
                # print(tmp[j:j+len(tmpx)].shape, j, tmpx, len(tmpx))
                tmp[j:j+len(tmpx)] = tmpx
                tmp_padded.append(tmp)
                new_y.append(y[i])
            tmp_padded = np.asarray(tmp_padded)
            if len(new_x) == 0:
                new_x = tmp_padded
            else:
                new_x = np.concatenate((new_x, tmp_padded), axis=0)
        return new_x, np.asarray(new_y)


def pad(x, symbol, max_length, mode = 'random'):

    if np.isscalar(x[0]):
        # print('scalr')
        # print(max_length - len(x))
        for j in range(max_length - len(x)):
            if mode == 'random':
                pos = np.random.randint(0, len(x) + 1, size = 1)[0]
                # print(pos)
                x = np.insert(x, pos, symbol)
            elif mode == 'append':
                x = np.insert(x, len(x), symbol)
            elif mode == 'splearn':
                if j == 0:
                    x = np.insert(x, 0, symbol)
                else:
                    x = np.insert(x, len(x), symbol)

    else:
        # print('array')
        for i in range(len(x)):
            # print(max_length - len(x[i]))
            for j in range(max_length - len(x[i])):
                if mode == 'random':
                    pos = np.random.randint(0, len(x[i]) + 1, size = 1)[0]
                    # print(symbol)
                    x[i] = np.insert(x[i], pos, symbol)
                    # print(x[i])
                elif mode == 'append':
                    x[i] = np.insert(x[i], len(x[i]), symbol)
                elif mode == 'splearn':
                    if j == 0:
                        x[i] = np.insert(x[i], 0, symbol)
                    else:
                        x[i] = np.insert(x[i], len(x[i]), symbol)

    return x
def filter_repetition(x, y):
    dict = {}
    new_x = []
    new_y = []
    for i in range(len(x)):
        if str(x[i]) not in dict:
            dict[str(x[i])] = 1
            new_x.append(x[i])
            new_y.append(y[i])
        else:
            continue
    # new_x = np.asarray(new_x)
    # new_y = np.asarray(new_y)
    return new_x, new_y

def get_hankel_tt_size(l, d, r):
    size = d*r*2 + (l-2)*d*r*r
    return size

def get_xy_from_splearn_factor(train_file, l, nbL, rank_tt = 12,
                               rank = 2, lrows=1, lcolumns=1, exact_length = False, mnne_factor = 0):
    train = spb.load_data_sample(train_file)
    est = splearn.Spectral(rank=rank, lrows=lrows, lcolumns=lcolumns, version='factor')

    train_dicts = est.polulate_dictionnaries(train.data)
    x = []
    y = []
    for key in train_dicts.fact.keys():
        x.append(list(key))
        y.append(train_dicts.fact[key])
    selected_indices = []
    for i in range(len(x)):
        if exact_length:
            if len(x[i]) == l:
                selected_indices.append(i)
        else:
            if len(x[i]) <= l:
                selected_indices.append(i)
    x = [x[index] for index in selected_indices]
    y = np.asarray([y[index] for index in selected_indices])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', len(x))
    # print(x)
    # print(y)
    tt_size = get_hankel_tt_size(l, nbL + 1, r=rank_tt)
    max_num_negative_examples = mnne_factor * (tt_size - len(x))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', len(x), nbL)
    x, y = add_negative_examples_v2(x, y, max_length=l, nbL=nbL, max_num_negative_examples=max_num_negative_examples)

    # x, y = add_negative_examples_v2(x, y, max_length=l, nbL=nbL, max_num_negative_examples=10*len(x))
    if not exact_length:
        new_x, new_y = pad_v2(copy.deepcopy(x), copy.deepcopy(y), -1, l, mode='splearn')
    else:
        new_x, new_y = x, y

    # new_y = np.asarray(copy.deepcopy(y))
    if not exact_length:
        x = one_hot(new_x, nbL)
    else:
        x = one_hot(new_x, nbL - 1)
    x = np.asarray(x)
    y = new_y

    x, y = filter_repetition(x, y)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', len(x))
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.arange(len(x))
    # idx = np.random.permutation(idx)
    x = x[idx]
    y = y[idx] * 1.
    return x, y



def get_xy_from_splearn(train_file, l, nbL, rank = 2, lrows=1, lcolumns=1, exact_length = False, rank_tt = 12, mnne_factor = 10):
    train = spb.load_data_sample(train_file)
    est = splearn.Spectral(rank=rank, lrows=lrows, lcolumns=lcolumns)
    train_dicts = est.polulate_dictionnaries(train.data)
    x = []
    y = []
    for key in train_dicts.sample.keys():
        x.append(list(key))
        y.append(train_dicts.sample[key])

    selected_indices = []
    for i in range(len(x)):
        # print(len(x[i]))
        if exact_length:
            if len(x[i]) == l:
                selected_indices.append(i)
        else:
            if len(x[i]) <= l:
                selected_indices.append(i)
    x = [x[index] for index in selected_indices]
    y = np.asarray([y[index] for index in selected_indices])
    tt_size = get_hankel_tt_size(l, nbL+1, r=rank_tt)
    max_num_negative_examples = mnne_factor * (tt_size - len(x))
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', len(x), nbL)
    x, y = add_negative_examples_v2(x, y, max_length=l, nbL=nbL, max_num_negative_examples= max_num_negative_examples)
    # print(x)
    # print(y)
    print('***************************************************', len(x))
    new_x, new_y = pad_v2(copy.deepcopy(x), copy.deepcopy(y), -1, l, mode='splearn')

    if exact_length:
        x = one_hot(new_x, nbL-1)
    else:
        x = one_hot(new_x, nbL)
    x = np.asarray(x)
    y = new_y

    x, y = filter_repetition(x, y)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', len(x))
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.arange(len(x))
    # idx = np.random.permutation(idx)
    x = x[idx]
    y = y[idx]*1.

    # print(x)
    # print(y)
    return x, y

def get_traj_up_to_L(prefixes, nbL, max_len = 10):
    all_traj = copy.deepcopy(prefixes)
    while len(prefixes[0]) < max_len:
        tmp = []
        for prefix  in prefixes:
            plus_one_trajs = get_traj_plus_one_len(prefix, nbL)
            tmp += plus_one_trajs
            all_traj+= plus_one_trajs
        prefixes = tmp
    return all_traj

def one_hot(x, nbL):
    # new_x = np.zeros([x.shape[0], x.shape[1],  nbL + 1])
    new_x = []
    for i in range(len(x)):
        tmp = np.zeros((len(x[i]), nbL + 1))
        # print(x[i])
        for j in range(len(x[i])):
            index = int(x[i][j])
            if index != -1:
                # new_x[i, j, index] = 1
                tmp[j][index] = 1
            else:
                tmp[j][-1] = 1

                # new_x[i, j, -1] = 1
        new_x.append(tmp)
    # new_x[:, -1, :] = 1.
    return new_x

def one_hot_no_pad(x, nbL):
    new_x = np.zeros([x.shape[0], x.shape[1],  nbL])
    for i in range(len(x)):
        for j in range(len(x[i])):
            index = int(x[i][j])
            new_x[i, j, index] = 1
    # new_x[:, -1, :] = 1.
    return new_x

def remove_lambda(core_list):
    new_core_list = []
    for core in core_list:
        new_core = core[:, :-1, :]
        new_core_list.append(new_core)
    return new_core_list

if __name__ == '__main__':
    # x = [[1, 2], [1, 2, 3], [1]]
    # print(pad_v2(x, symbol = -1, max_length = 4, mode = 'splearn'))

    train_file = '4.pautomac.train'
    test_file = '4.pautomac.test'
    solution_file = '4.pautomac_solution.txt'
    exact_length = False
    epochs = 1000

    l = 2
    pt = load_data_sample(train_file)
    # get_xy_from_splearn_factor(train_file, l, nbL = pt.nbL, rank=2, lrows=100, lcolumns=100, exact_length=True)
    get_xy_from_splearn(train_file, l, nbL=pt.nbL, rank=2, lrows=1, lcolumns=1, exact_length=exact_length)

