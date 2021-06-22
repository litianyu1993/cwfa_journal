import splearn
import splearn.datasets.base as spb
import numpy as np
def compute_perplexity(pred, y):
    tmp = 0.
    pred = np.abs(pred)
    for i in range(len(pred)):
        tmp += y[i]*np.log2(pred[i])
    return 2**(-tmp)
train_file = '4.pautomac.train'
test_file = '4.pautomac.test'
target_file = "4.pautomac_solution.txt"
targets =open(target_file, "r")
targets.readline()
target_proba = [float(line[:-1]) for line in targets]
target_proba = np.asarray(target_proba)
train = spb.load_data_sample(train_file)
test = spb.load_data_sample(test_file)
est = splearn.Spectral(rank = 12, lrows=3, lcolumns=3, version='factor')
est.fit(train.data)


print(np.max(est._hankel.lhankel[0].A))
U, s, V = np.linalg.svd(est._hankel.lhankel[0].A)
print(est.automaton.initial , est.automaton.final, est.automaton.transitions)
print(s)

pred = np.abs(est.predict(test.data)).reshape(target_proba.shape)
score = compute_perplexity(pred, target_proba)
print(score)

