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
#from utils.data_process import load_data

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
    r = net.fc2(r)
    r = net.bn(r)
    r /= np.sqrt(r.shape[1])
    r = F.relu(net.knn(r, alpha, r2, 3))
    r = r.detach().numpy()
    n = r.shape[0]
    nm = r.shape[1]
    # io.savemat("r.mat",dict([('r', r)]))
    # dx = np.zeros(n)
    # dmu = np.zeros(nm)
    count = r>0
    dx = np.sum(count, axis=1)
    dmu = np.sum(count, axis=0)
    return dx, dmu


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0, 1.1, generator=gen)

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

def pre_train(model, optimizer, dl_train, dl_test, x_train, y_train, x_validate, y_validate, run, outputDir,
              maxEpoch=200):
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
            PATH = outputDir + '/csnn_run{}_epoch{}.pth'.format(run, 0)
            torch.save({'net': model, 'alpha': alpha, 'r2': r2}, PATH)
            bestValidationAcc = accuracy

        if (epoch+1) % 100 == 0:
            print('validation: epoch {}, accuracy {:.4f}, loss {:.4f}'.format(epoch+1, accuracy, loss))
        losses_validate.append(loss)
        accuracies_validate.append(accuracy)
    plot_save_loss(losses, losses_validate, outputDir+'/loss_pretrain_run{}.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, outputDir+'/acc_pretrain_run{}.png'.format(run))
    print('Done pre-train, best validation acc {:.4f}'.format(bestValidationAcc))
    return accuracies, losses, accuracies_validate, losses_validate


class Net2(nn.Module):
    def __init__(self, inputs, features, bias=False):
        super().__init__()
        self.inputs = inputs
        self.fc1 = nn.Linear(inputs, features, bias=bias)
        self.fc4 = nn.Linear(features, 2, bias=False)
        # self.bn = nn.BatchNorm1d(features, affine=True)

    # def keepNodes(self, idx):
    #     # n=self.fc1.weight.shape[0]
    #     # idx=np.setdiff1d(np.arange(n),j)

    #     weight1 = self.fc1.weight.data[idx]
    #     weight2 = self.fc2.weight.data[:, idx]
    #     bias1 = self.fc1.bias.data[idx]
    #     # bias2 = self.fc2.bias.data

    #     # if 0:
    #     #     print(self.fc1.bias.shape)
    #     #     bias1 = self.fc1.bias.data[idx]
    #     #     bias2 = self.fc2.bias.data.clone()

    #     self.fc1 = nn.Linear(in_features=weight1.shape[1], out_features=weight1.shape[0])
    #     self.fc2 = nn.Linear(in_features=weight2.shape[1], out_features=weight2.shape[0], bias=False)

    #     self.fc1.weight.data = weight1
    #     self.fc1.bias.data = bias1
    #     self.fc2.weight.data = weight2

    #     # if 0:
    #     # self.fc1.bias.data = bias1
    #     # self.fc2.bias.data = bias2

    #     # print(self.fc2.weight.shape)

    # r^2-alpha x'x-mu'mu/alpha+2xmu
    # r^2=mu'mu/alpha+2b
    def knn(self, x, fc, alpha, r2, bias=False):
        xx = torch.mul(x, x)
        xx = torch.sum(xx, 1, keepdim=True)  # 1 for bias
        mm = torch.mul(fc.weight, fc.weight)
        # fc1 weight: outputDim x inputDim
        mm = torch.sum(mm, 1, keepdim=True).t()
        if bias:
            xx += 1
            mm += torch.mul(fc.bias, fc.bias).view(1, -1)
        x = alpha * (r2 - mm - xx) + 2 * fc(x)
        return x

    def forward(self, x, alpha, r2):
        x = F.relu(self.knn(x, self.fc1, alpha, r2))
        x = self.fc4(x)
        return x


class Net3(Net2):
    def __init__(self, inputs, features):
        super().__init__(inputs, features, True)
        self.inputs = inputs
        self.fc2 = nn.Linear(features, features, bias=False)
        # self.bn = nn.BatchNorm1d(features, affine=True)
        self.bn = nn.BatchNorm1d(features, affine=False)

    def forward(self, x, alpha, r2):
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = x / np.sqrt(x.shape[1])  # x /= sqrt(d)
        x = F.relu(self.knn(x, self.fc2, alpha, r2))
        x = self.fc4(x)
        return x


class Net4(Net2):
    def __init__(self, inputs, features):
        super().__init__(inputs, features, True)
        self.inputs = inputs
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features, bias=False)
        # self.bn = nn.BatchNorm1d(features, affine=True)
        self.bn = nn.BatchNorm1d(features, affine=False)

    def forward(self, x, alpha, r2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn(x)
        x = x / np.sqrt(x.shape[1])  # x /= sqrt(d)
        x = F.relu(self.knn(x, self.fc3, alpha, r2))
        x = self.fc4(x)
        return x


class Net(nn.Module):
    def __init__(self, inputs, features, bias=False):
        super().__init__()
        self.inputs = inputs
        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, features)
        self.fc4 = nn.Linear(features, 2, bias=False)
        # self.bn = nn.BatchNorm1d(features, affine=True)
        self.bn = nn.BatchNorm1d(features, affine=False)

    # def keepNodes(self, idx):
    #     # n=self.fc1.weight.shape[0]
    #     # idx=np.setdiff1d(np.arange(n),j)

    #     weight1 = self.fc1.weight.data[idx]
    #     weight2 = self.fc2.weight.data[:, idx]
    #     bias1 = self.fc1.bias.data[idx]
    #     # bias2 = self.fc2.bias.data

    #     # if 0:
    #     #     print(self.fc1.bias.shape)
    #     #     bias1 = self.fc1.bias.data[idx]
    #     #     bias2 = self.fc2.bias.data.clone()

    #     self.fc1 = nn.Linear(in_features=weight1.shape[1], out_features=weight1.shape[0])
    #     self.fc2 = nn.Linear(in_features=weight2.shape[1], out_features=weight2.shape[0], bias=False)

    #     self.fc1.weight.data = weight1
    #     self.fc1.bias.data = bias1
    #     self.fc2.weight.data = weight2

    #     # if 0:
    #     # self.fc1.bias.data = bias1
    #     # self.fc2.bias.data = bias2

    #     # print(self.fc2.weight.shape)

    # r^2-alpha x'x-mu'mu/alpha+2xmu
    # r^2=mu'mu/alpha+2b
    def knn(self, x, alpha, r2, layer):
        xx = torch.mul(x, x)
        xx = torch.sum(xx, 1, keepdim=True) + 1.  # 1 for bias
        if layer == 1:
            mm = torch.mul(self.fc1.weight, self.fc1.weight)
            # fc1 weight: outputDim x inputDim
            mm = torch.sum(mm, 1, keepdim=True).t()
            mm += torch.mul(self.fc1.bias, self.fc1.bias).view(1, -1)
            x = alpha * (r2 - mm - xx) + 2 * self.fc1(x)
        elif layer == 3:
            mm = torch.mul(self.fc3.weight, self.fc3.weight)
            # fc1 weight: outputDim x inputDim
            mm = torch.sum(mm, 1, keepdim=True).t()
            mm += torch.mul(self.fc3.bias, self.fc3.bias).view(1, -1)
            x = alpha * (r2 - mm - xx) + 2 * self.fc3(x)
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
        x = self.fc2(x)
        x = self.bn(x)
        x = x / np.sqrt(x.shape[1])  # x /= sqrt(d)
        x = F.relu(self.knn(x, alpha, r2, 3))
        x = self.fc4(x)
        return x


class csnn(nn.Module):
    def __init__(self, inputs, features, bias = False):
        super().__init__()
        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, 2, bias=False)
        # self.drop = nn.Dropout(p=dropoutRate)

    # r^2-alpha x'x-mu'mu/alpha+2xmu
    # r^2=mu'mu/alpha+2b
    def knn(self, x, alpha, r2):
        xx = torch.mul(x, x)
        xx = torch.sum(xx, 1, keepdim=True) + 1. # 1 for bias
        mm = torch.mul(self.fc1.weight, self.fc1.weight)
        # fc1 weight: outputDim x inputDim
        mm = torch.sum(mm, 1, keepdim=True).t()
        mm += torch.mul(self.fc1.bias, self.fc1.bias).view(1, -1)
        x = alpha * (r2 - mm - xx) + 2 * self.fc1(x)
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
        x = F.relu(self.knn(x, alpha, r2))
        x = self.fc2(x)
        return x
###


class MLP3(nn.Module):
    def __init__(self, inputs, features, bias=False):
        super().__init__()
        self.inputs = inputs
        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, 2)

    def forward(self, x, alpha, r2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP4(nn.Module):
    def __init__(self, inputs, features, bias=False):
        super().__init__()
        self.inputs = inputs
        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)
        self.fc4 = nn.Linear(features, 2)

    def forward(self, x, alpha, r2):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

