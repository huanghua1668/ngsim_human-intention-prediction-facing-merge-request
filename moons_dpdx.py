import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
import sklearn.datasets

import matplotlib.pyplot as plt
from matplotlib import cm

# import matplotlib.colors.Colormap as cmaps


class Model_bilinear(nn.Module):
    def __init__(self, features, num_embeddings):
        super().__init__()

        self.gamma = 0.99
        self.sigma = 0.3

        embedding_size = 10

        self.fc1 = nn.Linear(2, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)

        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1))
        self.register_buffer('N', torch.ones(num_embeddings) * 20)
        self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_embeddings), 1))
        ###
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        ###

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x size (64, 20) = batch_size, feature
        # i is batch, m is embedding_size, n is num_embeddings (classes)
        x = torch.einsum('ij,mnj->imn', x, self.W)

        return x

    def bilinear(self, z):
        # z is W_c.f(x)
        # embeddings is e_c
        embeddings = self.m / self.N.unsqueeze(0)
        # embeddings size (embedding_size, num_embeddings)

        diff = z - embeddings.unsqueeze(0)
        y_pred = (- diff ** 2).mean(1).div(2 * self.sigma ** 2).exp()

        return y_pred

    def forward(self, x):
        z = self.embed(x)
        y_pred = self.bilinear(z)

        return z, y_pred

    def update_embeddings(self, x, y):
        z = self.embed(x)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = torch.max(self.gamma * self.N + (1 - self.gamma) * y.sum(0), torch.ones_like(self.N))

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum('ijk,ik->jk', z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum


np.random.seed(0)
torch.manual_seed(0)

l_gradient_penalty = 1.0

# Moons
noise = 0.1
X_train, y_train = sklearn.datasets.make_moons(n_samples=1500, noise=noise)
X_test, y_test = sklearn.datasets.make_moons(n_samples=200, noise=noise)

num_classes = 2
batch_size = 64

model = Model_bilinear(20, num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


def calc_gradient_penalty(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        # retain_graph=True,
    )[0]
    # grad_outputs is the vector [[1],[1]] for matrix vector product
    # x (batch_size, 2), y_pred (batch_size, 2), gradients (batch_size, 2)
    # gradients = [d(y1+y2)/dx1, d(y1+y2)/dx2]
    gradients = gradients.flatten(start_dim=1)
    # no need to flatten here? as start_dim=1 for shape(batch_size, 2) will do nothing

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1.) ** 2).mean()
    # gradient_penalty = ((grad_norm** 2. - 1) ** 2).mean()
    # does not match eq.7 in paper, in which (grad_norm**2-1)**2

    # One sided penalty - down
    #     gradient_penalty = F.relu(grad_norm - 1).mean()

    return gradient_penalty

def cal_grad(x, y_pred):
    # x (batch_size, 2), y_pred (batch_size, 2)
    mask = torch.ones_like(y_pred)
    mask[:, 1] = 0
    dp1 = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=mask,
        retain_graph=True,
    )[0]
    mask = torch.ones_like(y_pred)
    mask[:, 0] = 0
    dp2 = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=mask,
        retain_graph=True,
    )[0]
    return dp1.detach().numpy(), dp2.detach().numpy()

def output_transform_acc(output):
    y_pred, y, x, z = output

    y = torch.argmax(y, dim=1)

    return y_pred, y


def output_transform_bce(output):
    y_pred, y, x, z = output

    return y_pred, y


def output_transform_gp(output):
    y_pred, y, x, z = output

    return x, y_pred


def step(model, optimizer, batch, l_gradient_penalty):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    z, y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    dp1, dp2 = cal_grad(x, y_pred)

    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    return loss.item(),x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy(), dp1, dp2


def eval_step(model, x):
    # optimizer.zero_grad()
    model.eval()

    x.requires_grad_(True)

    z, y_pred = model(x)
    dp1, dp2 = cal_grad(x, y_pred)

    # return y_pred, y, x, z
    return y_pred.detach().numpy(), dp1, dp2

def plot_concourf_full(ax, fig, mask, x, y, z, x0, y0, range, name, epoch):
    ax.tricontour(x, y, z, levels=14, colors='k')
    cntr = ax.tricontourf(x, y, z, range, cmap="RdBu_r", extend='both')
    fig.colorbar(cntr, ax=ax)
    ax.scatter(x0[mask], y0[mask], c='tab:blue', marker='o', s=2)
    ax.scatter(x0[~mask], y0[~mask], c='tab:orange', marker='o', s=2)
    ax.set(xlim=(-2.5, 3.5), ylim=(-3., 3.))
    ax.set_title(name + ' (epoch ' + str(epoch + 1) + ')')

def plot_concourf(ax, fig, mask, x, y, z, range, name, epoch):
    ax.tricontour(x, y, z, levels=14, colors='k')
    cntr = ax.tricontourf(x, y, z, range, cmap="RdBu_r", extend='both')
    fig.colorbar(cntr, ax=ax)
    ax.scatter(x[mask], y[mask], c='tab:blue', marker='o', s=2)
    ax.scatter(x[~mask], y[~mask], c='tab:orange', marker='o', s=2)
    ax.set(xlim=(-1.3, 2.3), ylim=(-0.8, 1.3))
    ax.set_title(name + ' (epoch ' + str(epoch + 1) + ')')

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                          F.one_hot(torch.from_numpy(y_train)).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=False)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), F.one_hot(torch.from_numpy(y_test)).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=200, shuffle=False)

for epoch in range(30):
    dp1s = []
    dp2s = []
    xs = []
    ys = []
    y_preds = []
    for i, batch in enumerate(dl_train):
        loss, x, y, y_pred, dp1, dp2 = step(model, optimizer, batch, l_gradient_penalty)
        dp1s.append(dp1)
        dp2s.append(dp2)
        xs.append(x)
        ys.append(y)
        y_preds.append(y_pred)
    x = np.vstack(xs)
    y = np.vstack(ys)
    y_pred = np.vstack(y_preds)
    dp1 = np.vstack(dp1s)
    dp2 = np.vstack(dp2s)
    dp1dx = dp1[:,0]
    dp1dy = dp1[:,1]
    dp2dx = dp2[:,0]
    dp2dy = dp2[:,1]
    mask = y[:,1]==1
    # if (epoch+1) % 5 == 0:
        # fig, ax = plt.subplots()
        # fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3)

        # range = np.linspace(-2.5, 2.5, 21)
        # plot_concourf(ax1, fig, mask, x[:,0], x[:,1], dp1dx, range, 'dp1dx', epoch)
        # plot_concourf(ax2, fig, mask, x[:,0], x[:,1], dp1dy, range, 'dp1dy', epoch)
        # plot_concourf(ax3, fig, mask, x[:,0], x[:,1], dp2dx, range, 'dp2dx', epoch)
        # plot_concourf(ax4, fig, mask, x[:,0], x[:,1], dp2dy, range, 'dp2dy', epoch)
        # plot_concourf(ax5, fig, mask, x[:,0], x[:,1], dp1dx+dp2dx, range, 'd(p1+p2)dx', epoch)
        # plot_concourf(ax6, fig, mask, x[:,0], x[:,1], dp1dy+dp2dy, range, 'd(p1+p2)dy', epoch)

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # range = np.linspace(0., 1., 21)
        # plot_concourf(ax1, fig, mask, x[:,0], x[:,1], y_pred[:,0], range, 'p1', epoch)
        # plot_concourf(ax2, fig, mask, x[:,0], x[:,1], y_pred[:,1], range, 'p2', epoch)
        # plot_concourf(ax3, fig, mask, x[:, 0], x[:, 1], np.sum(y_pred, axis=1), np.linspace(0.5, 1.5, 21), 'p1+p2', epoch)
    print('epoch ', epoch)

domain = 3
x_lin = np.linspace(-domain + 0.5, domain + 0.5, 100)
y_lin = np.linspace(-domain, domain, 100)
 # x_lin = np.linspace(-domain+0.5, domain+0.5, 200)
 # y_lin = np.linspace(-domain, domain, 200)

xx, yy = np.meshgrid(x_lin, y_lin)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
#
# X_vis, y_vis = sklearn.datasets.make_moons(n_samples=1000, noise=noise)
# mask = y_vis.astype(np.bool)
#
# with torch.no_grad():
y_pred, dp1, dp2 = eval_step(model, torch.from_numpy(X_grid).float())
dp1dx = dp1[:, 0]
dp1dy = dp1[:, 1]
dp2dx = dp2[:, 0]
dp2dy = dp2[:, 1]
# fig, ax = plt.subplots()
fig, ((ax1, ax3, ax5), (ax2, ax4, ax6)) = plt.subplots(2, 3)
range = np.linspace(-2.5, 2.5, 21)
mask = y[:, 1] == 1
plot_concourf_full(ax1, fig, mask, X_grid[:,0], X_grid[:,1], dp1dx, x[:,0], x[:,1], range, 'dp1dx', epoch)
plot_concourf_full(ax2, fig, mask, X_grid[:,0], X_grid[:,1], dp1dy, x[:,0], x[:,1], range, 'dp1dy', epoch)
plot_concourf_full(ax3, fig, mask, X_grid[:,0], X_grid[:,1], dp2dx, x[:,0], x[:,1], range, 'dp2dx', epoch)
plot_concourf_full(ax4, fig, mask, X_grid[:,0], X_grid[:,1], dp2dy, x[:,0], x[:,1], range, 'dp2dy', epoch)
plot_concourf_full(ax5, fig, mask, X_grid[:,0], X_grid[:,1], dp1dx+dp2dx, x[:,0], x[:,1], range, 'd(p1+p2)dx', epoch)
plot_concourf_full(ax6, fig, mask, X_grid[:,0], X_grid[:,1], dp1dy+dp2dy, x[:,0], x[:,1], range, 'd(p1+p2)dy', epoch)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
range = np.linspace(0., 1., 21)
plot_concourf_full(ax1, fig, mask, X_grid[:, 0], X_grid[:, 1], y_pred[:, 0], x[:,0], x[:,1], range, 'p1', epoch)
plot_concourf_full(ax2, fig, mask, X_grid[:, 0], X_grid[:, 1], y_pred[:, 1], x[:,0], x[:,1], range, 'p2', epoch)
plot_concourf_full(ax3, fig, mask, X_grid[:, 0], X_grid[:, 1], np.sum(y_pred, axis=1), x[:,0], x[:,1], np.linspace(0.5, 1.5, 21), 'p1+p2', epoch)
#
# z = confidence.reshape(xx.shape)
#
# plt.figure()
# # plt.contourf(x_lin, y_lin, z, cmap=cmaps.cividis)
# plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('inferno'))
# # plt.contourf(x_lin, y_lin, z, cmap=plt.get_cmap('cividis'))
# # plt.contourf(x_lin, y_lin, z)
# plt.scatter(X_vis[mask, 0], X_vis[mask, 1])
# plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1])
plt.show()
