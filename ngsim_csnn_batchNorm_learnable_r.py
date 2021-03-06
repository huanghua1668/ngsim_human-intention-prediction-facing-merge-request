import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from utils.plot_utils import plot_distribution
from utils.plot_utils import plot_save_loss
from utils.plot_utils import plot_save_acc
from utils.plot_utils import plot_save_acc_nzs_mmcs
from utils.plot_utils import plot_save_roc
from utils.data_preprocess import load_data

from models import nnz, active, step, eval_step, eval_combined, csnn, Net2, Net3, Net4_learnable_r, MLP3, MLP4, pre_train

# ngsim data
(x_train, y_train, x_validate, y_validate, x_ood) = load_data()
print('{} train samples, positive rate {:.3f}'.format(x_train.shape[0], np.mean(y_train)))
print('{} validate samples, positive rate {:.3f}'.format(x_validate.shape[0], np.mean(y_validate)))

dim = x_train.shape[1]
x_train = x_train/np.sqrt(dim)
x_validate = x_validate/np.sqrt(dim)
x_ood = x_ood/np.sqrt(dim)
x_combined = np.concatenate((x_validate, x_ood))

label_ood = np.zeros(x_combined.shape[0])
label_ood[x_validate.shape[0]:] = 1

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate).float(),
                                         F.one_hot(torch.from_numpy(y_validate)).float())

ds_combined = torch.utils.data.TensorDataset(torch.from_numpy(x_combined).float())

# network hyper param
inputs = 4
batchSize = 64
epochs = 200
hiddenUnits = 64
learningRate = 0.0004
# learningRate = 0.0001
l2Penalty = 1.0e-3
num_classes = 2

alpha = 0.
seeds = [0, 100057, 300089, 500069, 700079]
# runs = len(seeds)
runs = 5
r2 = 1.
maxAlpha = 1.

trained = True
bias = False
pretrained = True
outputDir='/home/hh/data/learnable_r/'

for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    model = Net4_learnable_r(inputs, hiddenUnits)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, x_validate, y_validate, run, outputDir, maxEpoch=10)

bestValidationAccs = []
AUCs = []
ACCs = []
epochs = 200
ALPHAs = None
for run in range(runs):
    np.random.seed(seeds[run])
    torch.manual_seed(seeds[run])
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
    PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
    l = torch.load(PATH)
    model = l['net']
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                 weight_decay=l2Penalty)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[50, 100, 150], gamma=0.5
    # )
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

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
        # alpha = min(1, max(0, (epoch**0.1-1.5)/0.6))
        # alpha = epoch/epochs
        # alpha = maxAlpha * epoch/epochs
        alpha = 0.3
        # r2 = min(0.01, 0.01 - epoch * 0.06 / 5000)
        # r2 = 1.
        for i, batch in enumerate(dl_train):
            loss_ce, loss_penalty, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r=True)

        accuracy, loss_ce, loss_penalty = eval_step(model, x_train, y_train, alpha, r2, learnable_r=True)
        #if epoch % 10 == 0:
        #    print('train: epoch {}, accuracy {:.3f}, loss {:.3f}'.format(epoch, accuracy, loss))
        testacc, testloss_ce, testloss_penalty = eval_step(model, x_validate, y_validate, alpha, r2, learnable_r=True)
        if testacc > bestValidationAcc:
            # include both learnable param and registered buffer
            # PATH = outputDir+'/csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, r2, maxAlpha)
            # torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
            bestValidationAcc = testacc

        if epoch % 5 == 0:
            #print('validation: epoch {}, fc1 shape {}, loss {:.3f}'.format(epoch, model.fc1.weight.data.shape[0], loss))
            losses.append(loss_ce+loss_penalty)
            losses_validate.append(testloss_ce+testloss_penalty)
            accuracies.append(accuracy)
            accuracies_validate.append(testacc)
            nz, mmc = nnz(x_ood, model, alpha, model.r * model.r)
            alphas.append(alpha)
            mmcs.append(mmc)
            nzs.append(nz)
            uncertainties = eval_combined(model, dl_combined, alpha, r2, learnable_r=True)
            falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
            AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
            aucs.append(AUC)
            rNorm = (torch.norm(model.r, p=2)).detach().numpy()
            print('epoch {}, alpha {:.2f}, r2 {:.1f}, nz {:.3f}, train {:.3f}, test {:.3f}, auroc {:.3f}, ||r||2 {:.3f}'
              .format(epoch, alpha, r2, 1.-nz,accuracy,testacc, AUC, rNorm))
            print('loss: cross_entropy {:.4f}, penalty {:.4f}'.format(loss_ce, loss_penalty))

        # eliminate dead nodes
        # if (   (epoch<200 and (epoch + 1) % (epochs / 100) == 0)
        #    or (epoch>=200 and (epoch + 1) % (epochs/10) == 0)):
        #    #_, dmu = active(torch.tensor(x_train).float(), model, alpha, r2)
        #    PATH = outputDir+'/csnn_2_csnn_layers_run{}_epoch{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, epoch+1, r2, maxAlpha)
        #    torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
        #    # model.keepNodes(dmu > 0)

    plot_save_loss(losses, losses_validate, outputDir+'/loss_run{}.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, outputDir+'/acc_run{}.png'.format(run))
    plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, aucs,
                           outputDir+'/acc_nzs_mmcs_run{}.png'.format(run))

    bestValidationAccs.append(max(accuracies_validate))
    AUCs.append(aucs)
    ACCs.append(accuracies_validate)
    if ALPHAs is None: ALPHAs = alphas
AUCs = np.array(AUCs)
ACCs = np.array(ACCs)
print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
      .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
dir = outputDir + '/mean_std_accs_aucs_net4.npz'
np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
              c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
              e=ALPHAs)
# plt.show()