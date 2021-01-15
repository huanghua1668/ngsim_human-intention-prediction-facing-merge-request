import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib.colors.Colormap as cmaps


class Model_bilinear(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)
        self.fc4 = nn.Linear(features, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.exp()
        return x


def step(batch, model, optimizer):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    # x.requires_grad_(True)

    z = model(x)
    y_pred = z/(z.sum(axis=1).view(-1,1))
    loss1 = F.binary_cross_entropy(y_pred, y)
    # loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)

    loss = loss1

    loss.backward()
    optimizer.step()

    # with torch.no_grad():
    #    model.update_embeddings(x, y)

    return loss.item()


def eval_step(model, batch):
    model.eval()

    x, y = batch

    # x.requires_grad_(True)
    with torch.no_grad():
        z = model(x)
    y_pred = z/(z.sum(axis=1).view(-1,1))
    return y.numpy(), z.detach().numpy(), y_pred.detach().numpy()


def eval_all(model, dl):
    zs = []
    ys = []
    y_preds = []
    for i, batch in enumerate(dl):
        y, z, y_pred = eval_step(model, batch)
        ys.append(y)
        zs.append(z)
        y_preds.append(y_pred)
    ys = np.vstack(ys)
    zs = np.vstack(zs)
    y_preds = np.vstack(y_preds)
    return ys, zs, y_preds


# Moons
noise = 0.1
batch_size = 64
X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
print('mean, std', mean, std)
X_train = (X_train-mean)/std/np.sqrt(2)
X_test, y_test = sklearn.datasets.make_moons(n_samples=200, noise=noise)
X_test = (X_test-mean)/std/np.sqrt(2)
ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

trained = False
domain = 3
x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
x_lin = (x_lin-mean[0])/std[0]/np.sqrt(2)
y_lin = (y_lin-mean[1])/std[1]/np.sqrt(2)
X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
X_vis = (X_vis-mean)/std/np.sqrt(2) # no need here, as contour grid is built on x_lin, y_lin
mask = y_vis.astype(np.bool)
if not trained:
    seeds = [100057, 300089, 500069, 700079, 900061, 1000081, 2000083, 3000073, 4000067, 5000101]
    models = []
    optimizers = []
    for i, seed in enumerate(seeds):
        print('model ', i)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = Model_bilinear(20)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        models.append(model)
        optimizers.append(optimizer)

        for epoch in range(30):
            for i, batch in enumerate(dl_train):
                step(batch, model, optimizer)
            y, z, y_pred = eval_all(model, dl_train)
            print('epoch {}, train acc {:.3f}'.format(epoch, np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))))
            y, z, y_pred = eval_all(model, dl_test)
            print('epoch {}, test acc {:.3f}'.format(epoch, np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))))

    # x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
    # y_lin = np.linspace(-domain, domain, 200)

    xx, yy = np.meshgrid(x_lin, y_lin)

    X_grid = np.column_stack([xx.flatten(), yy.flatten()])


    p = torch.zeros(X_grid.shape[0], 2)
    for model in models:
        with torch.no_grad():
            output = model(torch.from_numpy(X_grid).float())
            # confidence = output.max(1)[0].numpy()
            output /= output.sum(axis=1).view(-1,1)
            p += output
    p /= len(models)
    confidence = -(p * p.log()).sum(axis=1)

    z = confidence.reshape(xx.shape)
    outputDir =  '/home/hh/data/moons/deep_ensemble/'
    np.savez(outputDir + "confidence_map.npz", a=x_lin, b=y_lin, c=z)

outputDir = '/home/hh/data/moons/deep_ensemble/'
f= np.load(outputDir+"confidence_map.npz")
z = f['c']

plt.figure()
l = np.linspace(0, 1., 21)
# plt.contourf(x_lin, y_lin, z, cmap=cmaps.cividis)
# cntr = plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l, extend='both')
cntr = plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l)
plt.colorbar()
# plt.contourf(x_lin, y_lin, z)
# plt.contourf(x_lin, y_lin, z)
plt.scatter(X_vis[mask,0], X_vis[mask,1])
plt.scatter(X_vis[~mask,0], X_vis[~mask,1])
plt.show()