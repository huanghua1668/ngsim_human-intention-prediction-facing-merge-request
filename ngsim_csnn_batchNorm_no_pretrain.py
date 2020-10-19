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
from ngsim_deep_ensemble_no_bias_last_2_layers import Model_bilinear
from ngsim_duq import loadData
from utils.plot_utils import plot_save_roc
# import matplotlib.colors.Colormap as cmaps

###
def pdist2(x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    nmp1 = torch.stack([x] * ny).transpose(0, 1)
    nmp2 = torch.stack([y] * nx)
    return torch.sum((nmp1 - nmp2) ** 2, 2).squeeze()


def err(x, y, net, alpha, r2):
    wrong = 0
    total = 0
    with torch.no_grad():
        labels = torch.squeeze(y)
        outputs = net(x, alpha, r2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        wrong += (predicted != labels.long()).sum().item()
    return wrong * 1. / total


def nnz(x, net, alpha, r2):
    net.eval()
    '''count non-zero response nodes'''
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
    net.eval()
    with torch.no_grad():
        y = F.relu(net.fc1(x))
        y = F.relu(net.fc2(y))
        y = net.bn(y)
        r = F.relu(net.knn(y, alpha, r2))
        r = r.detach().numpy()
        # n = r.shape[0]
        # nm = r.shape[1]
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
        self.bn = nn.BatchNorm1d(features, affine=True)
        if bias:
            self.fc3 = nn.Linear(features, features)
            self.fc4 = nn.Linear(features, 2)
            self.bias = bias
        else:
            self.fc3 = nn.Linear(features, features, bias=False)
            self.fc4 = nn.Linear(features, 2, bias=False)
        # self.drop = nn.Dropout(p=dropoutRate)

    def keepNodes(self, idx):
        # n=self.fc1.weight.shape[0]
        # idx=np.setdiff1d(np.arange(n),j)

        weight1 = self.fc3.weight.data[idx]
        weight2 = self.fc4.weight.data[:, idx]
        # bias1 = self.fc3.bias.data[idx]
        # bias2 = self.fc4.bias.data

        # if 0:
        #     print(self.fc1.bias.shape)
        #     bias1 = self.fc1.bias.data[idx]
        #     bias2 = self.fc2.bias.data.clone()

        self.fc3 = nn.Linear(in_features=weight1.shape[1], out_features=weight1.shape[0])
        self.fc4 = nn.Linear(in_features=weight2.shape[1], out_features=weight2.shape[0])

        self.fc3.weight.data = weight1
        self.fc4.weight.data = weight2

        # if 0:
        # self.fc3.bias.data = bias1
        # self.fc4.bias.data = bias2

        # print(self.fc2.weight.shape)

    # r^2-alpha x'x-mu'mu/alpha+2xmu
    # r^2=mu'mu/alpha+2b
    def knn(self, x, alpha, r2):
        xx = torch.mul(x, x)
        xx = torch.sum(xx, 1, keepdim=True) #+ 1. # 1 for bias
        mm = torch.mul(self.fc3.weight, self.fc3.weight)
        # fc3 weight: outputDim x inputDim
        mm = torch.sum(mm, 1, keepdim=True).t()
        # mm += torch.mul(self.fc3.bias, self.fc3.bias).view(1, -1)
        x = alpha * (r2 - mm - xx) + 2 * self.fc3(x)
        return x

    # def getCircles(self, alpha):
    #     mu = self.fc1.weight.detach().numpy()
    #     b = 0  # self.fc1.bias.detach().numpy()
    #     r = np.sum(mu * mu, axis=1) / alpha + b
    #     return mu, r

    # def responseBits(self, x, alpha, r2):
    #     x = self.knn(x, alpha, r2).detach()
    #     x = (x > 0).long().numpy()
    #     return x

    def forward(self, x, alpha, r2):
        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.bn(x)
        # x = self.drop(x)
        x = F.relu(self.knn(x, alpha, r2))
        x = self.fc4(x)
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
            kernel_distance, _ = y_pred.max(1)
    return kernel_distance.numpy()


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
            PATH = '/home/hh/data/csnn_4_layers_batchNorm_run{}_epoch{}.pth'.format(run, 0)
            torch.save({'net': model, 'alpha': alpha, 'r2': r2}, PATH)
            bestValidationAcc = accuracy

        if (epoch+1) % 100 == 0:
            print('validation: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch+1, accuracy, loss))
        losses_validate.append(loss)
        accuracies_validate.append(accuracy)
    plot_save_loss(losses, losses_validate, '/home/hh/data/loss_csnn_pretrain_4_layers_batchNorm_run{}.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, '/home/hh/data/acc_csnn_pretrain_4_layers_batchNorm_run{}.png'.format(run))
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
    epochs = 100
    hiddenUnits = 64
    learningRate = 0.0004
    # learningRate = 0.0001
    dropoutRate = 0.3
    l2Penalty = 1.0e-3
    num_classes = 2

    # duq param
    alpha = 0.
    seeds = [0, 100057, 300089, 500069, 700079]
    # runs = len(seeds)
    runs = 5
    maxAlpha = 0.1

    # load parameters for g(x), two fc layer from deep ensemble
    # length_scales = np.array([0.4])
    # modelTrained = Model_bilinear(inputs, hiddenUnits, dropoutRate)
    # PATH = '/home/hh/data/deep_ensemble_run{}_ensemble{}_no_bias_last_2_layers.pth'.format(4, 2)
    # # best validation acc 0.8434 for this model
    # modelTrained.load_state_dict(torch.load(PATH))

    pretrained = True
    trained = True
    bias = False
    # trained = True
    # if not pretrained:
    #     for run in range(runs):
    #         np.random.seed(seeds[run])
    #         torch.manual_seed(seeds[run])
    #         dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    #         dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    #         dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
    #         model = Net(inputs, hiddenUnits, dropoutRate)
    #         optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
    #                                      weight_decay=l2Penalty)
    #         pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, run, maxEpoch=200)

    if not trained:
        bestValidationAccs = []
        for run in range(runs):
            np.random.seed(seeds[run])
            torch.manual_seed(seeds[run])
            model = Net(inputs, hiddenUnits, dropoutRate)
            # PATH = '/home/hh/data/csnn_4_layers_batchNorm_run{}_epoch{}.pth'.format(run, 0)
            # l = torch.load(PATH)
            # model = l['net']
            optimizer = torch.optim.Adam(model.parameters(), lr=learningRate,
                                         weight_decay=l2Penalty)
            # scheduler = torch.optim.lr_scheduler.MultiStepLR(
            #     optimizer, milestones=[50, 100, 150], gamma=0.5
            # )
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=False, drop_last=False)
            dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
            dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)

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
                r2 = 1.
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
                    # PATH = '/home/hh/data/csnn_4layers_batchNorm_run{}.pth'.format(run)
                    PATH = '/home/hh/data/csnn_4layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}.pth'.format(maxAlpha, run)
                    torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
                    bestValidationAcc = accuracy

                if (   (epoch<200 and (epoch + 1) % (epochs / 100) == 0)
                        or (epoch>=200 and (epoch + 1) % (epochs/10) == 0)):
                    # PATH = '/home/hh/data/csnn_4_layers_batchNorm_run{}_epoch{}.pth'.format(run, epoch + 1)
                    PATH = '/home/hh/data/csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}_epoch{}.pth'.format(maxAlpha, run, epoch + 1)
                    torch.save({'net': model, 'alpha': alpha, 'r2': r2}, PATH)

                if epoch % 50 == 0:
                    print('validation: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch, accuracy, loss))
                losses_validate.append(loss)
                accuracies_validate.append(accuracy)
                nz, mmc = nnz(x_ood, model, alpha, r2)
                alphas.append(alpha)
                mmcs.append(mmc)
                nzs.append(nz)
                if epoch % 50 == 0:
                    print('epoch {}, alpha {:.4f}, r2 {:.4f}, fc3 shape {}, num of zero z {:.4f}, mean max confidence {:.4f}'
                      .format(epoch, alpha, r2, model.fc3.weight.data.shape[0], 1.-nz, mmc))

                # eliminate dead nodes
                if epoch % (epochs/10) == 0:
                    _, dmu = active(torch.tensor(x_train).float(), model, alpha, r2)
                    print('fc3 non zero neurons {}'.format(np.sum(dmu>0)))
                    # model.keepNodes(dmu > 0)

            # print('epoch ', epoch)
            dir0 = '/home/hh/data/'
            dir = '/home/hh/data/loss_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}'.format(maxAlpha, run)
            np.savez(dir + ".npz", a=np.array(losses), b=np.array(losses_validate), c=np.array(mmcs))
            dir = '/home/hh/data/acc_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}'.format(maxAlpha, run)
            np.savez(dir + ".npz", a=np.array(accuracies), b=np.array(accuracies_validate))
            dir = '/home/hh/data/csnn_nz_mmcs_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}.npz'.format(maxAlpha, run)
            np.savez(dir, a=np.array(nzs), b=np.array(mmcs))

            plot_save_loss(losses, losses_validate,
                           '/home/hh/data/loss_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}.png'.format(maxAlpha, run))
            plot_save_acc(accuracies, accuracies_validate,
                           '/home/hh/data/acc_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}.png'.format(maxAlpha, run))
            plot_save_acc_nzs_mmcs(alphas, accuracies_validate, nzs, mmcs,
                           '/home/hh/data/acc_nzs_mmcs_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}.png'.format(maxAlpha, run))

            bestValidationAccs.append(max(accuracies_validate))

        print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
              .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
        # plt.show()
    else:
        AUCs = []
        ACCs = []
        epochs = []
        # for i in range(10, 191, 10):
        for i in range(10, 101, 10):
            epochs.append(i)
        # for i in range(200, 1001, 100):
        #     epochs.append(i)
        alphas = maxAlpha * np.array(epochs)/1000.
        for run in range(runs):
            aucs = []
            accs= []
            for epoch in epochs:
                # PATH = '/home/hh/data/csnn_4_layers_batchNorm_run{}_epoch{}.pth'.format(run, epoch)
                PATH = '/home/hh/data/csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}_epoch{}.pth'.format(maxAlpha, run, epoch)
                if epoch==0:
                    PATH = '/home/hh/data/csnn_4_layers_batchNorm_noPretrain_run{}_epoch{}.pth'.format(run, epoch)
                l = torch.load(PATH)
                model = l['net']
                alpha = l['alpha']
                r2 = l['r2']
                print('load model at run {}, alpha {:.4f}, r2 {:.4f}'.format(run, alpha, r2))
                dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
                uncertainties = eval_combined(model, dl_combined, alpha, r2)
                accuracy, loss = eval_step(model, x_validate, y_validate, alpha, r2)
                falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
                AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                aucs.append(AUC)
                accs.append(accuracy)
                print('run {}, epoch {}, acc {:.4f}, auc {:.4f}'.format(run, epoch, accuracy, AUC))

                dir = '/home/hh/data/roc_csnn_4_layers_batchNorm_noPretrain_maxAlpha{:.2f}_run{}_epoch{}.png'.format(maxAlpha, run, epoch)
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
        dir = '/home/hh/data/mean_std_accs_aucs_csnn_maxAlpha{:.2f}_4_layers_batchNorm_noPretrain.npz'.format(maxAlpha)
        np.savez(dir, a=np.mean(AUCs, axis=0), b=np.std(AUCs, axis=0),
                 c=np.mean(ACCs, axis=0), d=np.std(ACCs, axis=0),
                 e=alphas)


