import torch
import numpy as np
import pickle
from scipy.io import loadmat


def loadfile(pre_str, file_name):
    print(pre_str + file_name)
    f = loadmat(file_name)
    return f


dat_coll = {}
tr_coll = {}
ev_coll = {}

print("Start loading file")
for s in range(1, 4):
    dat_coll['s%d' % s] = {}
    for v in range(1, 3):
        d_name = "sub-%02d" % s
        f_name = "XS%02d_V%d.mat" % (s, v)

        f = loadfile("\t > ", 'DSC_2018.00114_120_v1/%s/%s' % (d_name, f_name))
        dat_coll['s%d' % s]['v%d' % v] = f['X'] if v == 1 else f['X_V2']

stim = loadfile("\t > ", 'DSC_2018.00114_120_v1/Y_brains.mat')['Y'].reshape(-1, 1, 56, 56)
dat_coll['y'] = np.transpose(stim, (0, 1, 3, 2))
dat_coll['l'] = loadfile("\t > ", 'DSC_2018.00114_120_v1/labels.mat')['L']

splitter = loadfile("Loading index splitter: ", 'DSC_2018.00114_120_v1/train_testset.mat')['trainidx']
splitter = (splitter - 1).squeeze(1)
sp_l = list(splitter)

print("Building train/eval collection...")
# Define training and eval masking
tr_m = np.array([False] * 360)
tr_m[sp_l] = True
ev_m = np.invert(tr_m)

for s in range(1, 4):
    tr_coll['s%d' % s] = {}
    ev_coll['s%d' % s] = {}

    for v in range(1, 3):
        d = dat_coll['s%d' % s]['v%d' % v]
        # fmri_data
        tr_coll['s%d' % s]['v%d' % v] = torch.Tensor(d[tr_m])
        ev_coll['s%d' % s]['v%d' % v] = torch.Tensor(d[ev_m])

tr_coll['y'] = torch.Tensor(dat_coll['y'][tr_m])
ev_coll['y'] = torch.Tensor(dat_coll['y'][ev_m])

tr_coll['l'] = torch.LongTensor(dat_coll['l'][tr_m]-1).squeeze(1)
ev_coll['l'] = torch.LongTensor(dat_coll['l'][ev_m]-1).squeeze(1)

pickle.dump(tr_coll, open('DSC_2018.00114_120_v1/fmriHC_train.dat', 'wb'))
pickle.dump(ev_coll, open('DSC_2018.00114_120_v1/fmriHC_eval.dat', 'wb'))

# import matplotlib.pyplot as plt
#
# for i in range(10):
#     test = tr_coll['y'][i, :, :, :].squeeze(0)
#     plt.imshow(test)
#     plt.show()
pass
