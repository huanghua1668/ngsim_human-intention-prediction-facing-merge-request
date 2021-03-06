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

class MLP(nn.Module):
    def __init__(self, input, features, dropoutRate):
        super().__init__()
        self.fc1 = nn.Linear(input, features)
        self.fc2 = nn.Linear(features, 2)
        self.drop = nn.Dropout(p=dropoutRate)
        self.m = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.m(x)
        # return x.exp()
        return x


def step(batch, model, optimizer):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    z = model(x)
    # p = z/(z.sum(axis=1).view(-1,1))
    # loss = F.binary_cross_entropy(p, y)
    loss = F.binary_cross_entropy(z, y)

    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(model, batch):
    model.eval()
    x, y = batch
    with torch.no_grad():
        z = model(x)
        # p = z/(z.sum(axis=1).view(-1,1))
    # return y.numpy(), p.detach().numpy()
    return y.numpy(), z.detach().numpy()


def eval_all(model, dl):
    ys = []
    y_preds = []
    for i, batch in enumerate(dl):
        y, y_pred = eval_step(model, batch)
        ys.append(y)
        y_preds.append(y_pred)
    ys = np.vstack(ys)
    y_preds = np.vstack(y_preds)
    return ys, y_preds


def plot_concourf_full(x, y, z, validationRange):
    fig, ax = plt.subplots()
    # ax.tricontour(x, y, z, levels=14, colors='k')
    range = np.linspace(-20., 0., 21)
    cntr = ax.tricontourf(x, y, z, range, cmap="RdBu_r", extend='both')
    fig.colorbar(cntr, ax=ax)
    ax.scatter(x[:validationRange], y[:validationRange], c='tab:blue', marker='o', s=2)
    ax.scatter(x[validationRange:], y[validationRange:], c='tab:orange', marker='o', s=2)
    # ax.set(xlim=(-2.5, 3.5), ylim=(-3., 3.))
    # ax.set_title(name + ' (epoch ' + str(epoch + 1) + ')')

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
    # x_ood = x_ood[::100]
    # x_ood = x_ood[::10]
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
    hiddenUnits = 512
    learningRate = 0.0004
    # learningRate = 0.0001
    dropoutRate = 0.3
    l2Penalty = 1.0e-3
    num_classes = 2


    # seeds = [0, 100057, 300089, 500069, 700079, 900061, 1000081, 2000083, 3000073, 4000067, 5000101]
    seed = 0
    runs = 1
    ensembles = 10
    models = []
    optimizers = []

    trained = False
    trained = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not trained:
        for i in range(runs):
            for j in range(ensembles):
                print('ensemble ', j)
                dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
                dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
                dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
                model = MLP(inputs, hiddenUnits, dropoutRate)
                optimizer = optim.Adam(model.parameters(), lr=learningRate,
                                       weight_decay=l2Penalty)
                losses = []
                accuracies = []
                losses_validate = []
                accuracies_validate = []
                aucs = []
                bestValidationAcc = 0.

                for epoch in range(epochs):
                    for k, batch in enumerate(dl_train):
                        step(batch, model, optimizer)
                    y, y_pred = eval_all(model, dl_train)
                    loss = F.binary_cross_entropy(torch.from_numpy(y_pred), torch.from_numpy(y))
                    accuracy = np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))
                    losses.append(loss)
                    accuracies.append(accuracy)
                    if epoch % 5 == 0:
                        print('epoch {}, train acc {:.3f}, loss {:.3f}'.format(epoch, accuracy, loss))
                    y, y_pred = eval_all(model, dl_test)
                    loss = F.binary_cross_entropy(torch.from_numpy(y_pred), torch.from_numpy(y))
                    accuracy = np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))
                    losses_validate.append(loss)
                    accuracies_validate.append(accuracy)
                    if epoch % 5 == 0:
                        print('epoch {}, test acc {:.3f}, loss {:.3f}'.format(epoch, accuracy, loss))
                        PATH = outputDir+'deep_ensemble_run{}_ensemble{}_epoch{}.pth'.format(i, j, epoch)
                        stateDict = model.state_dict()
                        torch.save(stateDict, PATH)
                    if accuracy > bestValidationAcc:
                        # include both learnable param and registered buffer
                        bestValidationAcc = accuracy
                # dir = '/home/hh/data/loss_deep_ensemble_run{}_ensemble{}.npz'.format(i, j)
                # np.savez(dir, a=np.array(losses), b=np.array(losses_validate))
                # dir = '/home/hh/data/acc_deep_ensemble_run{}_ensemble{}.npz'.format(i, j)
                # np.savez(dir, a=np.array(accuracies), b=np.array(accuracies_validate))

                plt.figure()
                plt.plot(np.arange(len(losses)) + 1, losses, label='train')
                plt.plot(np.arange(len(losses)) + 1, losses_validate, label='validate')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.axis([0, 200, 0.2, 1.8])
                plt.legend()
                dir = outputDir + 'loss_deep_ensemble_run{}_ensemble{}.png'.format(i, j)
                plt.savefig(dir)

                plt.figure()
                plt.plot(np.arange(len(losses)) + 1, accuracies, label='train')
                plt.plot(np.arange(len(losses)) + 1, accuracies_validate, label='validate')
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.axis([0, 200, 0.6, 1.])
                plt.legend()
                dir = outputDir+ 'acc_deep_ensemble_run{}_ensemble{}.png'.format(i, j)
                plt.savefig(dir)
    else:
        for i in range(runs):
            accs = []
            aucs = []
            for epoch in range(0, epochs, 5):
                exponents = []
                accuracies = []
                prob = np.zeros((x_combined.shape[0], 2))
                for j in range(ensembles):
                    model = MLP(inputs, hiddenUnits, dropoutRate)
                    PATH = outputDir + 'deep_ensemble_run{}_ensemble{}_epoch{}.pth'.format(i, j, epoch)
                    model.load_state_dict(torch.load(PATH))
                    y_pred = model(torch.from_numpy(x_combined).float())
                    exponents.append(y_pred)
                    prob += y_pred.detach().numpy()
                exponents = torch.stack(exponents)
                exponents = exponents.mean(0)
                # score, pred = exponents.max(1)
                _, pred = exponents.max(1)
                # prob /= ensembles
                # votes = votes > (ensembles//2)
                # votes = votes.astype(np.int)
                # accuracy = np.mean(prob[:x_validate.shape[0]].argmax(axis=1) == y_validate)
                # accuracy = np.mean(votes[:x_validate.shape[0]] == y_validate)
                accuracy = np.mean(pred[:x_validate.shape[0]].detach().numpy() == y_validate)
                accs.append(accuracy)
                # uncertainty = -np.sum(prob*np.log(prob), axis=1)
                # falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, uncertainty)
                score = exponents * torch.log(exponents)
                score = score.mean(1)
                falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -score.detach().numpy())
                AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                aucs.append(AUC)
                print('run {}, acc {:.4f}, auc {:.4f}'.format(i, accuracy, AUC))

            # plot_concourf_full(x_combined[:, 0], x_combined[:, 2], -score.detach().numpy(), x_validate.shape[0])
            # if i==0:
            #     scoreIn = -score[:x_validate.shape[0]].detach().numpy()
            #     scoreOut = -score[x_validate.shape[0]:].detach().numpy()
            #     plt.figure()
            #     # plt.figure()
            #     plt.hist(scoreOut, bins=50, range=(-1,-0.5), color='orange', label='OOD')
            #     plt.hist(scoreIn, bins=50, range=(-1,-0.5), color='blue', label='in-dis')
            #     plt.xlabel('score')
            #     plt.ylabel('Counts')
            #     plt.legend()
            #     plt.show()

            # plt.figure()
            # plt.plot(falsePositiveRate, truePositiveRate, color='darkorange',
            #          lw=2, label='ROC curve (auc = %0.4f)' % AUC)
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC')
            # plt.legend(loc="lower right")
            # dir = '/home/hh/data/roc_deep_ensemble_run{}.png'.format(i)
            # plt.savefig(dir)
        # print('average of AUC {:.4f}, std of of AUC {:.4f}'.format(np.mean(np.array(aucs)), np.std(np.array(aucs))))
