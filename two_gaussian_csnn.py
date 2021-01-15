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
from utils.plot_utils import plot_save_roc, plot_circles
from utils.data_preprocess import load_data_both_dataset
from models import nnz, active, step, eval_step, eval_combined, Net3, Net4, Net4_learnable_r, MLP3, MLP4, csnn, csnn_learnable_r, pre_train

def vis(model, X_grid, xx, x_lin, y_lin, X_vis, mask, epoch, dir0, alpha=1., learnable_r = False):
    with torch.no_grad():
        if learnable_r:
            output = model(torch.from_numpy(X_grid).float(), alpha, model.r * model.r)
        else:
            output = model(torch.from_numpy(X_grid).float(), alpha, 1.)
        output = F.softmax(output, dim=1)
        confidence = output.max(1)[0].numpy()

    z = confidence.reshape(xx.shape)

    plt.figure()
    l = np.linspace(0.5, 1., 21)
    plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l)  # , extend='both')
    plt.colorbar()
    X_vis = X_vis[::4]
    mask = mask[::4]
    plt.scatter(X_vis[mask, 0], X_vis[mask, 1])
    plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1])
    # plt.axis([-3, 3., -3, 3])
    dir = dir0 + '/confidence_epoch_{}.png'.format(epoch)
    plt.savefig(dir)
    # plt.show()

seeds = [0, 100057, 300089, 500069, 700079]
num_classes = 2
batchSize = 64
features = 64
# features = 256
learningRate = 0.001
l2Penalty = 1.0e-3
runs = 1
r2 = 1.
maxAlpha = 1.
LAMBDA = 1.28
MIU = 0.0
epochs = 500
outputDir='/home/hh/data/two_gaussian/'
learnable_r = True
BIAS = False

# Moons
noise = 0.1
# sklearn has no random seed, it depends on numpy to get random numbers

data = np.load(outputDir + 'two_gaussian_train_test.npz')
x_train = data['a']
y_train = x_train[:, -1]
y_train = y_train.astype(int)
x_train = x_train[:, :-1]
x_validate = data['b']
y_validate = x_validate[:, -1]
y_validate = y_validate.astype(int)
x_validate = x_validate[:, :-1]
x_train0 = x_train
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

print('mean, std', mean, std)
x_train = (x_train-mean)/std/np.sqrt(2)
x_validate = (x_validate-mean)/std/np.sqrt(2)

# dataset for image output
domain = 8
x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
y_lin = np.linspace(-domain, domain, 200)
# x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
# y_lin = np.linspace(-domain, domain, 200)
x_lin = (x_lin-mean[0])/std[0]/np.sqrt(2)
y_lin = (y_lin-mean[1])/std[1]/np.sqrt(2)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
# X_grid = (X_grid-mean)/std/np.sqrt(2)

X_vis, y_vis =  x_validate, y_validate
mask = y_vis.astype(np.bool)
# X_vis = (X_vis-mean)/std/np.sqrt(2) # no need here, as contour grid is built on x_lin, y_lin

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                         F.one_hot(torch.from_numpy(y_validate)).float())

accs = []
losses = []
accs_validate = []
losses_validate = []


# pre_train
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    # model = csnn(2, features)
    model = csnn_learnable_r(2, features, bias=BIAS)
    # model = Net4(2, features)
    # model = Net4_learnable_r(2, features)
    # model = MLP4(inputs, hiddenUnits)
    if learnable_r:
        model.set_lambda(LAMBDA)
        model.set_miu(MIU)
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
    rs = []

    bestValidationAcc = 0.
    np.set_printoptions(precision=4)
    for epoch in range(epochs):
        # alpha = maxAlpha * epoch/epochs
        alpha = maxAlpha
        for i, batch in enumerate(dl_train):
            if learnable_r:
                # loss_ce, loss_penalty, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r=True)
                loss_ce, loss_penalty, loss_l2, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r=True)
            else:
                loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

        if learnable_r:
            accuracy, loss_ce, loss_penalty, loss_l2 = eval_step(model, x_train, y_train, alpha, r2, learnable_r=True)
            testacc, testloss_ce, testloss_penalty, testloss_l2 = eval_step(model, x_validate, y_validate, alpha, r2, learnable_r=True)
        else:
            accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
            testacc, testloss = eval_step(model, x_validate, y_validate, alpha, r2)
        if testacc > bestValidationAcc:
            bestValidationAcc = testacc

        if epoch % 5 == 0:
            #print('validation: epoch {}, fc1 shape {}, loss {:.3f}'.format(epoch, model.fc1.weight.data.shape[0], loss))
            if learnable_r:
                losses.append(loss_ce+loss_penalty)
                losses_validate.append(testloss_ce+testloss_penalty)
            else:
                losses.append(loss)
                losses_validate.append(testloss)
            accuracies.append(accuracy)
            accuracies_validate.append(testacc)
            if learnable_r:
                rs.append([torch.norm(model.r, p=float('inf')).detach().item(), torch.norm(model.r, p=2).detach().item()])
                rNorm = (torch.norm(model.r, p=2)).detach().numpy()
                if BIAS:
                    w0 = model.fc1.bias.detach().numpy()
                    r0 = model.r.detach().numpy()
                    r0 = r0*r0 - (1-w0)*(1-w0)
                    r0[r0<0.] = 0.
                    r0 = np.sqrt(r0)
                    r0 = np.sort(r0)[-10:]
                r = np.sort(model.r.detach().numpy())[-10:]
                print('epoch {}, alpha {:.2f}, r2 {:.1f}, train {:.3f}, test {:.3f}, ||r|| {:.3f}'
                    .format(epoch, alpha, r2, accuracy,testacc, rNorm))
                # print('loss: cross_entropy {:.4f}, penalty {:.4f}'.format(loss_ce, loss_penalty))
                print('loss: cross_entropy {:.4f}, r penalty {:.4f}, w penalty {:.4f}'.format(loss_ce, loss_penalty, loss_l2))
                print('r top 10: ', r)
                if BIAS: print('true r top 10: ', r0)
            else:
                print('epoch {}, alpha {:.2f}, r2 {:.1f}, train {:.3f}, test {:.3f}'
                   .format(epoch, alpha, r2, accuracy, testacc))
        if epoch%10 == 0 or epoch<10:
            vis(model, X_grid, xx, x_lin, y_lin, X_vis, mask, epoch, outputDir, alpha, learnable_r)
            if BIAS: print('w0: ', np.sort(model.fc1.bias.detach().numpy())[::8])
            plot_circles(model.fc1, x_train, y_train, alpha, model.r.detach().numpy(), epoch, outputDir, BIAS)

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
         k=ALPHAs, l=np.array(rs))

