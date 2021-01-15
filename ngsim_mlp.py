import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.plot_utils import plot_distribution
from utils.plot_utils import plot_save_loss
from utils.plot_utils import plot_save_acc
from utils.plot_utils import plot_save_acc_nzs_mmcs
from utils.plot_utils import plot_save_roc
from utils.data_preprocess import load_data

from models import MLP3

def step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    z = model(x)
    y_pred = F.softmax(z, dim=1)
    loss_ce = F.binary_cross_entropy(y_pred, y)
    loss = loss_ce
    loss.backward()
    optimizer.step()
    return loss.item(), x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy(), z.detach().numpy()


def eval_step(model, x, y):
    model.eval()
    x = torch.tensor(x, dtype=torch.float)
    with torch.no_grad():
        z = model(x)
        y_pred = F.softmax(z, dim=1)
        y0 = np.copy(y)
        y = F.one_hot(torch.from_numpy(y)).float()
        loss = loss_ce = F.binary_cross_entropy(y_pred, y)
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        accuracy = np.mean(y0 == y_pred)
    return accuracy, loss.detach().item()

dir = '/home/hh/data/ngsim/'
f = np.load(dir + "combined_dataset.npz")
x_train = f['a']
y_train = f['b']
x_validate = f['c']
y_validate = f['d']
x_ood = f['e']
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
hiddenUnits = 512
learningRate = 0.001
# learningRate = 0.0001
l2Penalty = 1.0e-3

# seeds = [0, 100057, 300089, 500069, 700079]
seed = 0
# runs = len(seeds)
runs = 10
BIAS = False

trained = False
outputDir='/home/hh/data/ngsim/combined_dataset/MLP/'

bestValidationAccs = []
ACCs = []
epochs = 500
np.random.seed(seed)
torch.manual_seed(seed)
for run in range(runs):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batchSize, shuffle=True, drop_last=False)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=x_validate.shape[0], shuffle=False)
    model = MLP3(inputs, hiddenUnits, bias=BIAS)
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

    bestValidationAcc = 0.
    for epoch in range(epochs):
        for i, batch in enumerate(dl_train):
            # loss_ce, loss_r, loss_w, x, y, y_pred, z = step(model, optimizer, batch, alpha, r2, learnable_r)
            step(model, optimizer, batch)
        accuracy, loss = eval_step(model, x_train, y_train)
        testacc, testloss = eval_step(model, x_validate, y_validate)
        #if epoch % 10 == 0:
        #    print('train: epoch {}, accuracy {:.3f}, loss {:.3f}'.format(epoch, accuracy, loss))
        if testacc > bestValidationAcc:
            # include both learnable param and registered buffer
            # PATH = outputDir+'/csnn_2_csnn_layers_run{}_r2{:.1f}_maxAlpha{:.1f}_affine_false.pth'.format(run, r2, maxAlpha)
            # torch.save({'net':model, 'alpha':alpha, 'r2':r2}, PATH)
            bestValidationAcc = testacc

        if epoch % 5 == 0:
            #print('validation: epoch {}, fc1 shape {}, loss {:.3f}'.format(epoch, model.fc1.weight.data.shape[0], loss))

            losses.append(loss)
            losses_validate.append(testloss)
            accuracies.append(accuracy)
            accuracies_validate.append(testacc)
            print('epoch {}, train {:.3f}, test {:.3f}'.format(epoch,accuracy,testacc))

    plot_save_loss(losses, losses_validate, outputDir+'/loss_run{}.png'.format(run))
    plot_save_acc(accuracies, accuracies_validate, outputDir+'/acc_run{}.png'.format(run))
    bestValidationAccs.append(max(accuracies_validate))
    ACCs.append(accuracies_validate)

ACCs = np.array(ACCs)
print('mean and std of best validation acc in {} runs: {:.4f}, {:.4f}'
      .format(runs, np.mean(np.array(bestValidationAccs)), np.std(np.array(bestValidationAccs))))
dir = outputDir + '/mean_std_accs_aucs_net4.npz'
np.savez(dir, a=np.mean(ACCs, axis=0), b=np.std(ACCs, axis=0))
# plt.show()