import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.colors.Colormap as cmaps
# from ngsim_duq import loadData

if __name__ == "__main__":
    # ngsim data
    dir = '/home/hh/data/ngsim/'
    outputDir = '/home/hh/data/ngsim/combined_dataset/deep_ensemble/'
    f = np.load(dir + "combined_dataset.npz")
    x_train = f['a']
    y_train = f['b']
    x_validate = f['c']
    y_validate = f['d']
    x_ood = f['e']
    print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
    print('{} validate samples, positive rate {:.3f}'.format(x_validate.shape[0], np.mean(y_validate)))
    print('{} ood samples'.format(x_ood.shape[0]))
