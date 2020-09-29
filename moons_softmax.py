import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
# import matplotlib.colors.Colormap as cmaps
sns.set()


class Model_bilinear(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)
        self.fc4 = nn.Linear(features, 2)

        # self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1))
        # nn.parameter are automatically added to the list of its parameters, and will appear e.g. in
        # parameters() iterator. Assigning a Tensor doesn’t have such effect
        # print(list(model.parameters()))
        # self.register_buffer('N', torch.ones(num_embeddings) * 20)
        # self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_embeddings), 1))
        ###
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        ###

        # self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x size (64, 20) = batch_size, feature
        # i is batch, m is embedding_size, n is num_embeddings (classes)
        #x = torch.einsum('ij,mnj->imn', x, self.W)
        x = self.fc4(x)
        x = x.exp()
        return x

    # def bilinear(self, z):
    #     # z is W_c.f(x)
    #     # embeddings is e_c
    #     embeddings = self.m / self.N.unsqueeze(0)
    #     # embeddings size (embedding_size, num_embeddings)

    #     diff = z - embeddings.unsqueeze(0)
    #     y_pred = (- diff ** 2).mean(1).div(2 * self.sigma ** 2).exp()
    #     # y_pred = (- diff ** 2).mean(1).exp()

    #     return y_pred

    def forward(self, x):
        z = self.embed(x)
        # y_pred = self.bilinear(z)

        return z

    # def update_embeddings(self, x, y):
    #     z = self.embed(x)

    #     # normalizing value per class, assumes y is one_hot encoded
    #     self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))

    #     # compute sum of embeddings on class by class basis
    #     features_sum = torch.einsum('ijk,ik->jk', z, y)

    #     self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

# Moons
noise = 0.1
X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
X_test, y_test = sklearn.datasets.make_moons(n_samples=200, noise=noise)

num_classes = 2
batch_size = 64

model = Model_bilinear(20)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


# def calc_gradient_penalty(x, y_pred):
#     gradients = torch.autograd.grad(
#         outputs=y_pred,
#         inputs=x,
#         grad_outputs=torch.ones_like(y_pred),
#         create_graph=True
#         # retain_graph=True,
#     )[0]
#     ###
#      # With create_graph=True, we are declaring that we want to do further operations on gradients, so that the autograd
#      # engine can create a backpropable graph for operations done on gradients.
#     ###
#     # direct output is a list of len 1, so output [0]
#     # x (batch_size, 2), y_pred (batch_size, 2), gradients (batch_size, 2)
#     # gradients = [d(y1+y2)/dx1, d(y1+y2)/dx2]
#     # gradients = gradients.flatten(start_dim=1)
#     # no need to flatten here? as start_dim=1 for shape(batch_size, 2) will do nothing
#
#     # L2 norm
#     grad_norm = gradients.norm(2, dim=1)
#
#     # Two sided penalty
#     gradient_penalty = ((grad_norm - 1.) ** 2).mean()
#     # gradient_penalty = ((grad_norm** 2. - 1) ** 2).mean()
#     # does not match eq.7 in paper, in which (grad_norm**2-1)**2
#
#     # One sided penalty - down
#     #     gradient_penalty = F.relu(grad_norm - 1).mean()
#
#     return gradient_penalty


# def output_transform_acc(output):
#     y_pred, y, x, z = output
#
#     y = torch.argmax(y, dim=1)
#
#     return y_pred, y
#
#
# def output_transform_bce(output):
#     y_pred, y, x, z = output
#
#     return y_pred, y
#
#
# def output_transform_gp(output):
#     y_pred, y, x, z = output
#
#     return x, y_pred


def step(batch):
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

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)


for epoch in range(30):
    for i, batch in enumerate(dl_train):
        step(batch)
    y, z, y_pred = eval_all(model, dl_train)
    print('epoch {}, train acc {:.3f}'.format(epoch, np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))))
    y, z, y_pred = eval_all(model, dl_test)
    print('epoch {}, test acc {:.3f}'.format(epoch, np.mean(y_pred.argmax(axis=1)==y.argmax(axis=1))))

domain = 3
x_lin = np.linspace(-domain+0.5, domain+0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
# x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
# y_lin = np.linspace(-domain, domain, 200)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])

X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
mask = y_vis.astype(np.bool)

with torch.no_grad():
    output = model(torch.from_numpy(X_grid).float())
    # confidence = output.max(1)[0].numpy()
    output /= output.sum(axis=1).view(-1,1)
    confidence = (output * output.log()).sum(axis=1)

z = confidence.reshape(xx.shape)

plt.figure()
l = np.linspace(-0.7, 0., 15)
# plt.contourf(x_lin, y_lin, z, cmap=cmaps.cividis)
cntr = plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'), levels=l, extend='both')
plt.colorbar()
# plt.contourf(x_lin, y_lin, z)
# plt.contourf(x_lin, y_lin, z)
plt.scatter(X_vis[mask,0], X_vis[mask,1])
plt.scatter(X_vis[~mask,0], X_vis[~mask,1])
plt.show()