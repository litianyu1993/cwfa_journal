import pickle
import numpy as np
from ALS_CWFA import tt_to_tensor, set_to_zero
version = 'factor'
with open('true_hankel' + version, 'rb') as f:
    true_hankel = pickle.load(f)
    true_hl, true_h2l, true_h2l1 = true_hankel[0], true_hankel[1], true_hankel[2]
    # print(true_hl)
    # print(true_h2l)
    # print(true_h2l1)

with open('sgdv2_h2l'+ version, 'rb') as f:
    H_2l = set_to_zero(tt_to_tensor(pickle.load(f)))
with open('sgdv2_h2l1'+ version, 'rb') as f:
    H_2l1 = set_to_zero(tt_to_tensor(pickle.load(f)))
with open('sgdv2_hl'+ version, 'rb') as f:
    H_l = set_to_zero(tt_to_tensor(pickle.load(f)))


print(H_l - true_hl)
print(H_l)
print(true_hl)
print(np.mean((H_l - true_hl)**2))
print(np.mean((H_2l - true_h2l)**2))
print(np.mean((H_2l1 - true_h2l1)**2))