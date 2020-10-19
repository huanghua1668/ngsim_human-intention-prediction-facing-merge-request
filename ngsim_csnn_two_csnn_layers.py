import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from utils.plot_utils import plot_distribution
from utils.plot_utils import plot_save_loss
from utils.plot_utils import plot_save_acc
from utils.plot_utils import plot_save_acc_nzs_mmcs
from utils.plot_utils import plot_save_roc
from ngsim_deep_ensemble_no_bias_last_2_layers import Model_bilinear
from ngsim_duq import loadData
# import matplotlib.colors.Colormap as cmaps

###
# def pdist2(x, y):
#     nx = x.shape[0]
#     ny = y.shape[0]
#     nmp1 = torch.stack([x] * ny).transpose(0, 1)
#     nmp2 = torch.stack([y] * nx)
#     return torch.sum((nmp1 - nmp2) ** 2, 2).squeeze()
#
#
# def err(x, y, net, alpha, r2):
#     wrong = 0
#     total = 0
#     with torch.no_grad():
#         labels = torch.squeeze(y)
#         outputs = net(x, alpha, r2)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         wrong += (predicted != labels.long()).sum().item()
#     return wrong * 1. / total


def nnz(x, net, alpha, r2):
    '''count non-zero response nodes'''
    net.eval()
    nz = 0
    total = 0
    mmc = 0
    with torch.no_grad():
        outputs = net(torch.tensor(x).float(), alpha, r2)
        m, _ = torch.max(torch.abs(outputs), 1)
        # print('m values for ood: ', m[:-1:1000])
        # (values, indices)
        total += x.shape[0]
        nz += (m != 0).sum().item()
        sm = F.softmax(outputs, 1)
        mmci, _ = torch.max(sm, 1)
        mmc += mmci.sum().item()
        # mean max confidence
    return nz * 1. / total, mmc / total


def active(x, net, alpha, r2):
    r = F.relu(net.knn(x, alpha, r2, 1))
    r = net.bn(r)
    r = F.relu(net.knn(r, alpha, r2, 2))
    r = r.detach().numpy()
    n = r.shape[0]
    nm = r.shape[1]
    # io.savemat("r.mat",dict([('r', r)]))
    # dx = np.zeros(n)
    # dmu = np.zeros(nm)
    count = r>0
    dx = np.sum(count, axis=1)
    dmu = np.sum(count, axis=0)
    # for i in range(n):
    #     dx[i] = np.sum(r[i] > 0)
    #     # get how many elements>0 in each row
    # for i in range(nm):
    #     # get how many elements>0 in each col
    #     dmu[i] = np.sum(r[:, i] > 0)
    return dx, dmu


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0, 1.1, generator=gen)


class Net(nn.Module):
    def __init__(self, inputs, features, dropoutRate, bias = False):
        super().__init__()
        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, 2, bias=False)
        # self.bn = nn.BatchNorm1d(features, affine=True)
        self.bn = nn.BatchNorm1d(features, affine=False)
        # self.drop = nn.Dropout(p=dropoutRate)

    def keepNodes(self, idx):
        # n=self.fc1.weight.shape[0]
        # idx=np.setdiff1d(np.arange(n),j)

        weight1 = self.fc1.weight.data[idx]
        weight2 = self.fc2.weight.data[:, idx]
        bias1 = self.fc1.bias.data[idx]
        # bias2 = self.fc2.bias.data

        # if 0:
        #     print(self.fc1.bias.shape)
        #     bias1 = self.fc1.bias.data[idx]
        #     bias2 = self.fc2.bias.data.clone()

        self.fc1 = nn.Linear(in_features=weight1.shape[1], out_features=weight1.shape[0])
        self.fc2 = nn.Linear(in_features=weight2.shape[1], out_features=weight2.shape[0], bias=False)

        self.fc1.weight.data = weight1
        self.fc1.bias.data = bias1
        self.fc2.weight.data = weight2

        # if 0:
        # self.fc1.bias.data = bias1
        # self.fc2.bias.data = bias2

        # print(self.fc2.weight.shape)

    # r^2-alpha x'x-mu'mu/alpha+2xmu
    # r^2=mu'mu/alpha+2b
    def knn(self, x, alpha, r2, layer):
        xx = torch.mul(x, x)
        xx = torch.sum(xx, 1, keepdim=True) + 1. # 1 for bias
        if layer == 1:
            mm = torch.mul(self.fc1.weight, self.fc1.weight)
            # fc1 weight: outputDim x inputDim
            mm = torch.sum(mm, 1, keepdim=True).t()
            mm += torch.mul(self.fc1.bias, self.fc1.bias).view(1, -1)
            x = alpha * (r2 - mm - xx) + 2 * self.fc1(x)
        elif layer == 2:
            mm = torch.mul(self.fc2.weight, self.fc2.weight)
            # fc1 weight: outputDim x inputDim
            mm = torch.sum(mm, 1, keepdim=True).t()
            mm += torch.mul(self.fc2.bias, self.fc2.bias).view(1, -1)
            x = alpha * (r2 - mm - xx) + 2 * self.fc2(x)
        return x

    # def getCircles(self, alpha):
    #     mu = self.fc1.weight.detach().numpy()
    #     b = 0  # self.fc1.bias.detach().numpy()
    #     r = np.sum(mu * mu, axis=1) / alpha + b
    #     return mu, r

    # def responseBits(self, x, alpha, r2):
    #     x = self.knn(x, alpha, r2, 1).detach()
    #     x = (x > 0).long().numpy()
    #     return x

    def forward(self, x, alpha, r2):
        x = F.relu(self.knn(x, alpha, r2, 1))
        x = self.bn(x)
        x = F.relu(self.knn(x, alpha, r2, 2))
        x = self.fc3(x)
        return x
###


def step(model, optimizer, batch, alpha, r2):
    model.train()
    optimizer.zero_grad()

    x, y = batch

    z = model(x, alpha, r2)
    y_pred = F.softmax(z, dim=1)
    loss = F.binary_cross_entropy(y_pred, y)

    loss.backward()
    optimizer.step()

    return loss.item(), x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy(), z.detach().numpy()


def eval_step(model, x, y, alpha, r2):
    model.eval()
    x = torch.tensor(x, dtype=torch.float)
    with torch.no_grad():
        z = model(x, alpha, r2)
        y_pred = F.softmax(z, dim=1)
        y0 = np.copy(y)
        y = F.one_hot(torch.from_numpy(y)).float()
        loss = F.binary_cross_entropy(y_pred, y)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        accuracy = np.mean(y0 == y_pred)
    return accuracy, loss.detach().item()


# accuracy, bce_loss, gp_loss = eval_combined(model, dl_combined)
def eval_combined(model, dl_combined, alpha, r2):
    model.eval()
    for i, batch in enumerate(dl_combined):
        x = batch[0]
        with torch.no_grad():
            z = model(x, alpha, r2)
    y_pred = F.softmax(z, dim=1)
    confidence, _ = y_pred.max(1)
    return confidence.numpy()


def pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, run, maxEpoch=200):
    losses = []
    accuracies = []
    losses_validate = []
    accuracies_validate = []

    bestValidationAcc = 0.
    alpha = 0.
    r2 = 0.
    for epoch in range(maxEpoch):
        for i, batch in enumerate(dl_train):
            loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

        accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
        losses.append(loss)
        accuracies.append(accuracy)
        if (epoch+1) % 100 == 0:
            print('train: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch+1, accuracy, loss))
        accuracy, loss = eval_step(model, x_validate, y_validate, alpha, r2)
        if accuracy > bestValidationAcc:
            # stateDict = model.state_dict()
            # include both learnable param and registered buffer
            # PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}.pth'.format(run, 0)
            PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}_affine_false.pth'.format(run, 0)
            torch.save({'net': model, 'alpha': alpha, 'r2': r2}, PATH)
            bestValidationAcc = accuracy

        if (epoch+1) % 100 == 0:
            print('validation: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch+1, accuracy, loss))
        losses_validate.append(loss)
        accuracies_validate.append(accuracy)
    plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_pretrain_2_csnn_layers_run{}_affine_false.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_pretrain_2_csnn_layers_run{}_affine_false.png'.format(run))
    print('Done pre-train, best validation acc {:.4f}'.format(bestValidationAcc))

if __name__ == "__main__":
    # ngsim data
    (x_train0, x_train, y_train, x_validate0, x_validate, y_validate, x_ood) = loadData()
    # x_ood = x_ood[::100]
    # x_ood = x_ood[::10]
    y_train[y_train == -1] = 0
    y_validate[y_validate == -1] = 0
    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    x_train = x_train[:, mask]
    x_validate = x_validate[:, mask]
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
    epochs = 1000
    hiddenUnits = 64
    learningRate = 0.0004
    # learningRate = 0.0001
    dropoutRate = 0.3
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
    # trained = True
    if not pretrained:
        for run in range(runs):
            np.random.seed(seeds[run])
            torch.manual_seed(seeds[run])
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
            dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
            dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
            model = Net(inputs, hiddenUnits, dropoutRate)
            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                         weight_decay=l2Penalty)
            pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, run, maxEpoch=200)

    if not trained:
        bestValidationAccs = []
        for run in range(runs):
            np.random.seed(seeds[run])
            torch.manual_seed(seeds[run])
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
            dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
            dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
            PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}_affine_false.pth'.format(run, 0)
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
                alpha = maxAlpha * epoch/epochs
                # r2 = min(0.01, 0.01 - epoch * 0.06 / 5000)
                # r2 = 1.
                for i, batch in enumerate(dl_train):
                    loss, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2)

                accuracy, loss = eval_step(model, x_train, y_train, alpha, r2)
                losses.append(loss)
                accuracies.append(accuracy)
                if epoch % 50 == 0:
                    print('train: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch, accuracy, loss))
                accuracy, loss = eval_step(model, x_validate, y_validate, alpha, r2)
                if accuracy > bestValidationAcc:
                    # stateDict = model.state_dict()
                    # include both learnable param and registered buffer
                    # PATH = '/home/hh/data/csnn_2_layers_run{}.pth'.format(run)
                    # PATH = '/home/hh/data/csnn_2_layers_run{}_trim.pth'.format(run)
                    # PATH = '/home/hh/data/csnn_2_layers_run{}_r2{:.1f}.pth'.format(run, r2)
                    PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, r2, maxAlpha)
                    torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
                    bestValidationAcc = accuracy

                if epoch % 50 == 0:
                    print('validation: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch, accuracy, loss))
                losses_validate.append(loss)
                accuracies_validate.append(accuracy)
                nz, mmc = nnz(x_ood, model, alpha, r2)
                alphas.append(alpha)
                mmcs.append(mmc)
                nzs.append(nz)
                if epoch % 50 == 0:
                    print('epoch {}, alpha {:.4f}, r2 {:.4f}, fc1 shape {}, num of zero z {:.4f}, mean max confidence {:.4f}'
                      .format(epoch, alpha, r2, model.fc1.weight.data.shape[0], 1.-nz, mmc))

                # eliminate dead nodes
                if (   (epoch<200 and (epoch + 1) % (epochs / 100) == 0)
                    or (epoch>=200 and (epoch + 1) % (epochs/10) == 0)):
                    _, dmu = active(torch.tensor(x_train).float(), model, alpha, r2)
                    print('fc1 non zero neurons {}'.format(np.sum(dmu>0)))
                    # PATH = '/home/hh/data/csnn_2_layers_run{}_epoch{}.pth'.format(run, epoch+1)
                    # PATH = '/home/hh/data/csnn_2_layers_run{}_epoch{}_trim.pth'.format(run, epoch+1)
                    # PATH = '/home/hh/data/csnn_2_layers_run{}_epoch{}_r2{:.1f}.pth'.format(run, epoch+1, r2)
                    PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, epoch+1, r2, maxAlpha)
                    torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
                    # model.keepNodes(dmu > 0)

            # print('epoch ', epoch)

            # dir = '/home/hh/data/loss_csnn_2_layers_run{}'.format(run)
            # dir = '/home/hh/data/loss_csnn_2_layers_run{}_trim'.format(run)
            # dir = '/home/hh/data/loss_csnn_2_layers_run{}_r2{:.1f}'.format(run, r2)
            dir = '/home/hh/data/loss_csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false'.format(run, r2, maxAlpha)
            np.savez(dir + ".npz", a=np.array(losses), b=np.array(losses_validate), c=np.array(mmcs))
            # dir = '/home/hh/data/acc_csnn_2_layers_run{}'.format(run)
            # dir = '/home/hh/data/acc_csnn_2_layers_run{}_trim'.format(run)
            # dir = '/home/hh/data/acc_csnn_2_layers_run{}_r2{:.1f}'.format(run, r2)
            dir = '/home/hh/data/acc_csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false'.format(run, r2, maxAlpha)
            np.savez(dir + ".npz", a=np.array(accuracies), b=np.array(accuracies_validate))
            # dir = '/home/hh/data/csnn_nz_mmcs_2_layers_run{}.npz'.format(run)
            #dir = '/home/hh/data/csnn_nz_mmcs_2_layers_run{}_trim.npz'.format(run)
            # dir = '/home/hh/data/csnn_nz_mmcs_2_layers_run{}_r2{:.1f}.npz'.format(run, r2)
            dir = '/home/hh/data/csnn_nz_mmcs_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.npz'.format(run, r2, maxAlpha)
            np.savez(dir, a=np.array(nzs), b=np.array(mmcs))

            # plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_2_layers_run{}.png'.format(run))
            # plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_2_layers_run{}.png'.format(run))
            # plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, mmcs,
            #                        '/home/hh/data/acc_nzs_mmcs_csnn_2_layers_run{}.png'.format(run))
            # plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_2_layers_run{}_trim.png'.format(run))
            # plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_2_layers_run{}_trim.png'.format(run))
            # plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, mmcs,
            #                        '/home/hh/data/acc_nzs_mmcs_csnn_2_layers_run{}_trim.png'.format(run))
            # plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_2_layers_run{}_r2{:.1f}.png'.format(run, r2))
            # plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_2_layers_run{}_r2{:.1f}.png'.format(run, r2))
            # plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, mmcs,
            #                        '/home/hh/data/acc_nzs_mmcs_csnn_2_layers_run{}_r2{:.1f}.png'.format(run, r2))
            plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.png'.format(run, r2, maxAlpha))
            plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.png'.format(run, r2, maxAlpha))
            plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, mmcs,
                                   '/home/hh/data/acc_nzs_mmcs_csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.png'.format(run, r2, maxAlpha))

            bestValidationAccs.append(max(accuracies_validate))

        print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
              .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
        # plt.show()
    else:
        AUCs = []
        ACCs = []
        epochs = []
        for i in range(0, 191, 10):
            epochs.append(i)
        for i in range(200, 1001, 100):
            epochs.append(i)
        alphas = maxAlpha * np.array(epochs)/1000.
        dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
        for run in range(runs):
            aucs = []
            accs= []
            for epoch in epochs:
                # PATH = '/home/hh/data/csnn_2_layers_run{}_epoch{}_trim.pth'.format(run, epoch)
                # PATH = '/home/hh/data/csnn_2_layers_run{}_epoch{}_r2{:.1f}.pth'.format(run, epoch, r2)
                PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, epoch, r2, maxAlpha)
                if epoch==0:
                    PATH = '/home/hh/data/csnn_2_csnn_layers_run{}_epoch{}_affine_false.pth'.format(run, epoch)
                l = torch.load(PATH)
                print('load model at run {}, epoch {}'.format(run, epoch))
                model = l['net']
                alpha = l['alpha']
                # r2 = l['r2']
                print('load model at run {}, epoch {}, alpha {:.4f}, r2 {:.4f}'.format(run, epoch, alpha, r2))
                uncertainties = eval_combined(model, dl_combined, alpha, r2)
                accuracy, loss = eval_step(model, x_validate, y_validate, alpha, r2)
                accs.append(accuracy)
                falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
                AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                aucs.append(AUC)
                print('run {}, epoch {}, acc {:.4f}, auc {:.4f}'.format(run, epoch, accuracy, AUC))
                # dir = '/home/hh/data/roc_csnn_2_layers_run{}_epoch{}_trim.png'.format(run, epoch)
                # dir = '/home/hh/data/roc_csnn_2_layers_run{}_epoch{}_r2{:.1f}.png'.format(run, epoch, r2)
                dir = '/home/hh/data/roc_csnn_2_csnn_layers_run{}_epoch{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.png'.format(run, epoch, r2, maxAlpha)
                plot_save_roc(falsePositiveRate, truePositiveRate, AUC, dir)
            AUCs.append(aucs)
            ACCs.append(accs)
        AUCs = np.array(AUCs)
        ACCs = np.array(ACCs)
        print('average and std of AUC in {} runs'.format(runs))
        print('mean ', np.mean(AUCs, axis=0))
        print('std ', np.std(AUCs, axis=0))
        # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_trim.npz'
        # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r2{:.1f}.npz'.format(r2)
        dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_csnn_layers_r2{:.1f}_maxAlpha{:.1f}_affine_false.npz'.format(r2, maxAlpha)
        np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
                      c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
                      e=alphas)


