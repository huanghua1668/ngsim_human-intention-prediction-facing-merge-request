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
# import matplotlib.colors.Colormap as cmaps


class Model_bilinear(nn.Module):
    def __init__(self, inputs, features, num_embeddings, sigma, dropoutRate, gamma=0.99, embedding_size=10, nInit=20):
        # features are number of hidden units...
        super().__init__()

        self.gamma = gamma
        self.sigma = sigma

        self.fc1 = nn.Linear(inputs, features)
        self.fc2 = nn.Linear(features, features)
        self.fc3 = nn.Linear(features, features)
        self.drop = nn.Dropout(p=dropoutRate)
        self.bn1 = nn.BatchNorm1d(features)
        self.bn2 = nn.BatchNorm1d(features)
        self.bn3 = nn.BatchNorm1d(features)

        self.W = nn.Parameter(torch.normal(torch.zeros(embedding_size, num_embeddings, features), 1))
        self.register_buffer('N', torch.ones(num_embeddings) * nInit)
        self.register_buffer('m', torch.normal(torch.zeros(embedding_size, num_embeddings), 1))
        ###
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        ###

        self.m = self.m * self.N.unsqueeze(0)

    def embed(self, x):
        batchNorm = False
        if batchNorm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.bn3(self.fc3(x))
        else:
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = F.relu(self.fc2(x))
            x = self.drop(x)
            x = self.fc3(x)
        # x = self.drop(x)
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

def loadData():
    dir = '/home/hh/ngsim/I-80-Emeryville-CA/i-80-vehicle-trajectory-data/vehicle-trajectory-data/'
    f = np.load(dir + 'train_normalized.npz')
    x_train = f['a']
    y_train = f['b']
    print('x_train', x_train.shape[0], y_train.mean(), 'coop rate')
    y_train[y_train == 0] = -1

    f = np.load(dir + 'validate_normalized.npz')
    x_validate = f['a']
    y_validate = f['b']
    print('x_validate', x_validate.shape[0], y_validate.mean(), 'coop rate')
    y_validate[y_validate == 0] = -1

    f = np.load(dir + 'train_origin.npz')
    x_train0 = f['a']
    print(x_train0.shape)

    f = np.load(dir + 'validate_origin.npz')
    x_validate0 = f['a']
    print(x_validate0.shape)

    f = np.load('/home/hh/data/ood_sample.npz')
    x_ood = f['a']
    print(x_ood.shape)
    return (x_train0, x_train, y_train, x_validate0, x_validate, y_validate, x_ood)


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


def step(model, optimizer, batch, l_gradient_penalty):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x.requires_grad_(True)

    z, y_pred = model(x)

    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    # dp1, dp2 = cal_grad(x, y_pred)

    loss = loss1 + loss2

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.update_embeddings(x, y)

    # return loss.item(),x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy(), dp1, dp2
    return loss.item(),x.detach().numpy(), y.detach().numpy(), y_pred.detach().numpy()


def eval_step(model, x, y, l_gradient_penalty):
    model.eval()

    # x, y = batch
    x = torch.tensor(x, dtype=torch.float)
    x.requires_grad_(True)

    z, y_pred = model(x)
    y0 = np.copy(y)
    y = F.one_hot(torch.from_numpy(y)).float()
    loss1 = F.binary_cross_entropy(y_pred, y)
    loss2 = l_gradient_penalty * calc_gradient_penalty(x, y_pred)
    loss = loss1 + loss2
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    accuracy = np.mean(y0 == y_pred)
    return accuracy, loss1.detach().numpy(), loss2.detach().numpy()


# accuracy, bce_loss, gp_loss = eval_combined(model, dl_combined)
def eval_combined(model, dl_combined):
    model.eval()
    for i, batch in enumerate(dl_combined):
        x = batch[0]
        with torch.no_grad():
            z, y_pred = model(x)
    kernel_distance, pred = y_pred.max(1)
    return kernel_distance.numpy()


if __name__ == "__main__":
    # ngsim data
    (x_train0, x_train, y_train, x_validate0, x_validate, y_validate, x_ood) = loadData()
    # x_ood = x_ood[::100]
    # x_ood = x_ood[::10]
    y_train[y_train == -1] = 0
    y_validate[y_validate == -1] = 0
    mask = np.array([1, 2, 4, 5]) # already been chosen to delete
    x_combined = np.concatenate((x_validate[:, mask], x_ood))
    label_ood = np.zeros(x_combined.shape[0])
    label_ood[x_validate.shape[0]:] = 1

    ds_train = torch.utils.data.TensorDataset(torch.from_numpy(x_train[:, mask]).float(),
                                              F.one_hot(torch.from_numpy(y_train)).float())

    ds_test = torch.utils.data.TensorDataset(torch.from_numpy(x_validate[:, mask]).float(),
                                             F.one_hot(torch.from_numpy(y_validate)).float())

    ds_combined = torch.utils.data.TensorDataset(torch.from_numpy(x_combined).float())

    # network hyper param
    inputs = 4
    batchSize = 64
    epochs = 200
    hiddenUnits = 64
    learningRate = 0.0004
    # learningRate = 0.0001
    dropoutRate = 0.3
    l2Penalty = 1.0e-3
    num_classes = 2

    # duq param
    lambdas = np.linspace(0., 1., 11)
    lambdas = np.round(lambdas, 1)
    # length_scales = np.linspace(0.1, 1., 10)
    length_scales = np.array([0.4])
    seeds = [0, 100057, 300089, 500069, 700079]
    # runs = len(seeds)
    runs = 5

    # trained = False
    trained = True
    if not trained:
        for k in range(lambdas.shape[0]):
            for j in range(length_scales.shape[0]):
                bestValidationAccs = []
                for run in range(runs):
                    np.random.seed(seeds[run])
                    torch.manual_seed(seeds[run])
                    model = Model_bilinear(inputs, hiddenUnits, num_classes, length_scales[j], dropoutRate)
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
                    aucs = []
                    bestValidationAcc = 0.
                    for epoch in range(epochs):
                        for i, batch in enumerate(dl_train):
                            loss, x, y, y_pred = step(model, optimizer, batch, lambdas[k])
                        accuracy, bce_loss, gp_loss = eval_step(model, x_train[:, mask], y_train, lambdas[k])
                        losses.append(bce_loss + gp_loss)
                        accuracies.append(accuracy)
                        if epoch % 50 == 0:
                            print('train: epoch', epoch, ', bce loss', bce_loss, 'gp loss', gp_loss, 'accuracy', accuracy)
                        accuracy, bce_loss, gp_loss = eval_step(model, x_validate[:, mask], y_validate, lambdas[k])
                        if accuracy > bestValidationAcc:
                            stateDict = model.state_dict()
                            # include both learnable param and registered buffer
                            PATH = '/home/hh/data/lambda'+str(lambdas[k])+'_sigma' + str(length_scales[j]) + '_runs' + str(run)+'.pth'
                            torch.save(stateDict, PATH)
                            bestValidationAcc = accuracy

                        if epoch % 50 == 0:
                            print('validation: epoch', epoch, ', bce loss', bce_loss, 'gp loss', gp_loss, 'accuracy', accuracy)
                        losses_validate.append(bce_loss + gp_loss)
                        accuracies_validate.append(accuracy)
                        # print('epoch ', epoch)

                    dir = '/home/hh/data/loss_lambda'+str(lambdas[k])+'_sigma'+str(length_scales[j])+'_runs'+str(run)
                    np.savez(dir + ".npz", a=np.array(losses), b=np.array(losses_validate))
                    dir = '/home/hh/data/acc_lambda'+str(lambdas[k])+'_sigma'+str(length_scales[j])+'_runs'+str(run)
                    np.savez(dir + ".npz", a=np.array(accuracies), b=np.array(accuracies_validate))

                    plt.figure()
                    plt.plot(np.arange(len(losses)) + 1, losses, label='train')
                    plt.plot(np.arange(len(losses)) + 1, losses_validate, label='validate')
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.axis([0, 200, 0.2, 1.8])
                    plt.legend()
                    dir = '/home/hh/data/loss_lambda'+str(lambdas[k])+'_sigma'+str(length_scales[j])+'_runs'+str(run)
                    plt.savefig(dir+'.png')

                    plt.figure()
                    plt.plot(np.arange(len(losses)) + 1, accuracies, label='train')
                    plt.plot(np.arange(len(losses)) + 1, accuracies_validate, label='validate')
                    plt.xlabel('epoch')
                    plt.ylabel('accuracy')
                    plt.axis([0, 200, 0.6, 0.85])
                    plt.legend()
                    dir = '/home/hh/data/acc_lambda'+str(lambdas[k])+'_sigma'+str(length_scales[j])+'_runs'+str(run)
                    plt.savefig(dir+'.png')

                    bestValidationAccs.append(max(accuracies_validate))

                print('lambda ', lambdas[k], ', sigma ', length_scales[j],
                      ', average best validation acc in', runs, ' runs: ', np.mean(np.array(bestValidationAccs)),
                      ', std of best validation acc in', runs, ' runs: ', np.std(np.array(bestValidationAccs)))
                # plt.show()
    else:
        for k in range(lambdas.shape[0]):
            for j in range(length_scales.shape[0]):
                AUCs = []
                for run in range(runs):
                    model = Model_bilinear(inputs, hiddenUnits, num_classes, length_scales[j], dropoutRate)
                    PATH = '/home/hh/data/lambda' + str(lambdas[k]) + '_sigma' + str(length_scales[j]) + '_runs' + str(
                        run) + '.pth'
                    model.load_state_dict(torch.load(PATH))
                    dl_combined = torch.utils.data.DataLoader(ds_combined, batch_size=x_combined.shape[0], shuffle=False)
                    uncertainties = eval_combined(model, dl_combined)
                    if k==lambdas.shape[0]-1 and run==runs-1:
                        plot_distribution(-uncertainties, x_validate.shape[0])
                    falsePositiveRate, truePositiveRate, _= roc_curve(label_ood, -uncertainties)
                    AUC = auc(falsePositiveRate.astype(np.float32), truePositiveRate.astype(np.float32))
                    AUCs.append(AUC)
                    print(k, j, 'load parameters successfully, auc ', AUC)

                    plt.figure()
                    plt.plot(falsePositiveRate, truePositiveRate, color='darkorange',
                             lw=2, label='ROC curve (auc = %0.4f)' % AUC)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC')
                    plt.legend(loc="lower right")
                    dir = '/home/hh/data/roc_lambda'+str(lambdas[k])+'_sigma'+str(length_scales[j])+'_runs'+str(run)
                    plt.savefig(dir+'.png')
                print('lambda ', lambdas[k], ', sigma ', length_scales[j],
                      ', average AUC in', runs, ' runs: ', np.mean(np.array(AUCs)),
                      ', std of AUC in', runs, ' runs: ', np.std(np.array(AUCs)))


