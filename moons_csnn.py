import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.colors.Colormap as cmaps

from utils.plot_utils import plot_distribution
from utils.plot_utils import plot_save_loss
from utils.plot_utils import plot_save_acc
from utils.plot_utils import plot_save_acc_nzs_mmcs
from utils.plot_utils import plot_save_roc
from utils.data_preprocess import load_data_both_dataset
from models import nnz, active, step, eval_step, eval_combined, Net3, Net4, MLP3, MLP4, csnn, pre_train

# Moons
noise = 0.1
x_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
x_validate, y_validate = sklearn.datasets.make_moons(n_samples=200, noise=noise)
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
print('mean, std', mean, std)
x_train = (x_train-mean)/std/np.sqrt(2)
x_validate = (x_validate-mean)/std/np.sqrt(2)

seeds = [0, 100057, 300089, 500069, 700079]
num_classes = 2
batchSize = 64
features = 256
learningRate = 0.0005
l2Penalty = 1.0e-3

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                         F.one_hot(torch.from_numpy(y_validate)).float())

accs = []
losses = []
accs_validate = []
losses_validate = []

runs = 1
r2 = 1.
maxAlpha = 1.

outputDir='/home/hh/data/moons/'

# pre_train
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    # model = csnn(2, features)
    model = Net4(2, features)
    # model = MLP4(inputs, hiddenUnits)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    accuracy, loss, accuracy_validate, loss_validate = pre_train(model, optimizer, dl_train, dl_test, x_train,
                                                                 y_train, x_validate, y_validate, run, outputDir,
                                                                 maxEpoch=5)
    accs.append(accuracy)
    losses.append(loss)
    accs_validate.append(accuracy_validate)
    losses_validate.append(loss_validate)
dir = outputDir + 'pre_train_acc_loss_csnn.npz'
np.savez(dir, a=np.mean(accs, axis=0), b=np.std(accs, axis=0),
         c=np.mean(losses, axis=0), d=np.std(losses, axis=0),
         e=np.mean(accs_validate, axis=0), f=np.std(accs_validate, axis=0),
         g=np.mean(losses_validate, axis=0), h=np.std(losses_validate, axis=0))

ACCs = []
ACCs_val = []
LOSSs = []
LOSSs_val = []
epochs = 100
ALPHAs = None
bestValidationAccs = []
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
    l = torch.load(PATH)
    model = l['net']
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)

    losses = []
    accuracies = []
    losses_validate = []
    accuracies_validate = []
    alphas = []
    mmcs = []
    nzs = []
    aucs = []

    bestValidationAcc = 0.
    for epoch in range(epochs):
        alpha = maxAlpha * epoch/epochs
        for i, batch in enumerate(dl_train):
            loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

        accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
        testacc, testloss = eval_step(model, x_validate, y_validate, alpha, r2)
        if testacc > bestValidationAcc:
            bestValidationAcc = testacc

        if epoch % 5 == 0:
            #print('validation: epoch {}, fc1 shape {}, loss {:.3f}'.format(epoch, model.fc1.weight.data.shape[0], loss))
            losses.append(loss)
            losses_validate.append(testloss)
            accuracies.append(accuracy)
            accuracies_validate.append(testacc)
            # nz, mmc = nnz(x_ood, model, alpha, r2)
            # alphas.append(alpha)
            # mmcs.append(mmc)
            # nzs.append(nz)
            # uncertainties = eval_combined(model, dl_combined, alpha, r2)
            # falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
            # AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
            # aucs.append(AUC)
            # print('epoch {}, alpha {:.2f}, r2 {:.1f}, nz {:.3f}, train {:.3f}, test {:.3f}, auroc {:.3f}'
            #       .format(epoch, alpha, r2, 1.-nz,accuracy,testacc, AUC))
            print('epoch {}, alpha {:.2f}, r2 {:.1f}, train {:.3f}, test {:.3f}'
                  .format(epoch, alpha, r2, accuracy,testacc))

    plot_save_loss(losses, losses_validate, outputDir+'/loss_run{}.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, outputDir+'/acc_run{}.png'.format(run))
    # plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, aucs,
    #                        outputDir+'/acc_nzs_mmcs_run{}.png'.format(run))

    bestValidationAccs.append(max(accuracies_validate))
    # AUCs.append(aucs)
    ACCs.append(accuracies)
    ACCs_val.append(accuracies_validate)
    LOSSs.append(losses)
    LOSSs_val.append(losses_validate)
    if ALPHAs is None: ALPHAs = alphas

# AUCs = np.array(AUCs)
ACCs = np.array(ACCs)
ACCs_val = np.array(ACCs_val)
LOSSs = np.array(LOSSs)
LOSSs_val = np.array(LOSSs_val)
print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
      .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
dir = outputDir + '/mean_std_accs_aucs_net4.npz'
np.savez(dir,# a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
         c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
         e=np.mean(ACCs_val, axis=0), f=np.std(ACCs_val, axis=0),
         g=np.mean(LOSSs, axis=0), h=np.std(LOSSs, axis=0),
         i=np.mean(LOSSs_val, axis=0), j=np.std(LOSSs_val, axis=0),
         k=ALPHAs)

# plt.show()
domain = 3
x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
# x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
# y_lin = np.linspace(-domain, domain, 200)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
X_grid = (X_grid-mean)/std/np.sqrt(2)

X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
mask = y_vis.astype(np.bool)

with torch.no_grad():
    output = model(torch.from_numpy(X_grid).float(), 1., 1.)
    output = F.softmax(output, dim=1)
    confidence = output.max(1)[0].numpy()


z = confidence.reshape(xx.shape)

plt.figure()
# plt.contourf(x_lin, y_lin, z, cmap=cmaps.cividis)
plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'))
# plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('cividis'))
# plt.contourf(x_lin, y_lin, z)
plt.scatter(X_vis[mask,0], X_vis[mask,1])
plt.scatter(X_vis[~mask,0], X_vis[~mask,1])
plt.show()