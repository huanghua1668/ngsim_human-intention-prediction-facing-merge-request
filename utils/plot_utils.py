import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib._color_data as mcd
import matplotlib.cm as cm

def plot_loss_functions():
    def logistic(x):
        return -np.log(1./(1.+np.exp(-x)))

    def svm(x):
        y=1.-x
        y[y<0]=0.
        return y

    def lorenz(x):
        y=np.log(1.+(x-1.)*(x-1.))
        y[x>1.]=0.
        return y

    plt.figure()
    x=np.arange(-5,4)
    plt.plot(x,logistic(x), label='logistic loss')
    plt.plot(x,svm(x), label='hinge loss')
    plt.plot(x,lorenz(x), label='lorenz')
    plt.legend()
    plt.axis([-5, 3, -0.5, 6])

    plt.show()

def plot_feature_selections():
    validation_acc = [0.846, 0.846, 0.8484, 0.846, 0.8434, 0.8434, 0.8409, 0.8333, 0.8080]
    features = [10, 9, 8, 7, 6, 5, 4, 3, 2]
    plt.figure()
    plt.plot(features, validation_acc, '-o')
    plt.axis([10, 0, 0.8, 0.85])
    plt.ylabel('validation acc')
    plt.xlabel('features')
    plt.show()


def plot_save_loss(losses, losses_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(losses)) + 1, losses, label='train')
    plt.plot(np.arange(len(losses)) + 1, losses_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, 200, 0.2, 0.8])
    plt.legend()
    plt.savefig(dir)


def plot_save_acc(accuracies, accuracies_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(accuracies)) + 1, accuracies, label='train')
    plt.plot(np.arange(len(accuracies)) + 1, accuracies_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0, 200, 0.6, 0.95])
    plt.legend()
    plt.savefig(dir)


def plot_save_acc_average_std():
    outputDir = '/home/hh/data/ngsim/combined_dataset/MLP/'
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    f = np.load(dir)
    avg = f['a']
    std = f['b']
    data = np.concatenate((avg, std))
    np.savetxt(outputDir+'acc_mean_std.csv', data)
    plt.figure()
    plt.plot(5*np.arange(avg.shape[0]), avg)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0, 500, 0.6, 0.95])
    plt.savefig(outputDir+'acc.png')

def plot_save_acc_nzs_mmcs(alphas, acc, nzs, mmcs, dir):
    plt.figure()
    plt.plot(alphas, acc, label='ACC')
    plt.plot(alphas, nzs, label='NZ')
    plt.plot(alphas, mmcs, label='MMC')
    plt.xlabel('$\\alpha$')
    plt.ylabel('ACC, NZ, MMC')
    plt.axis([0, 1., 0., 1.])
    plt.legend()
    plt.savefig(dir)


def plot_auc():
    auc = np.round(np.loadtxt('../auc_sigma0.4.dat'), 4)
    acc = np.round(np.loadtxt('../acc_sigma0.4.dat'), 4)
    output = []
    for i in range(11):
        temp = [acc[i,0], acc[i,1], auc[i,1]]
        print(temp)
        output.append(temp)
    output = np.array(output)
    np.savetxt('../acc_auc.csv', output, fmt='%1.4f')

    plt.figure()
    plt.plot(auc[:,0], auc[:,1], '-o')
    plt.axis([0, 1.0, 0.9, 1.])
    plt.ylabel('average AUC')
    plt.xlabel('$\lambda $')

    plt.figure()
    plt.plot(acc[:,0], acc[:,1], '-o')
    plt.axis([0, 1.0, 0.8, 0.85])
    plt.ylabel('average best validation accuracy')
    plt.xlabel('$\lambda $')
    plt.show()


def plot_distribution(score, nValidation):
    scoreIn = score[:nValidation]
    scoreOut = score[nValidation:][::100]
    plt.figure()
    plt.hist(scoreOut, bins=50, range=(-1, -0.), color='orange', label='OOD')
    plt.hist(scoreIn, bins=50, range=(-1, -0.), color='blue', label='in-dis')
    plt.xlabel('score')
    plt.ylabel('Counts')
    plt.legend()
    dir = '/home/hh/data/score_distribution_duq.png'
    plt.savefig(dir)

def plot_func():
    epochs = 1000
    delta =200
    x=np.arange(epochs)
    y=np.exp((x-epochs)/delta)
    plt.figure()
    plt.plot(x, y)
    plt.show()

def plot_save_roc(falsePositiveRate, truePositiveRate, AUC, dir):
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
    plt.savefig(dir)

def plot_auc_acc_csnn():

    LAMBDAS = [0., 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
               1.6, 1.7, 1.8, 1.9, 2.]
    outputDir = '/home/hh/data/ngsim/combined_dataset/'
    for LAMBDA in LAMBDAS:
        dir = outputDir + '/mean_std_accs_aucs_net4_lambda{:.2f}.npz'.format(LAMBDA)
        f = np.load(dir)
        avg_auc = f['a'].reshape(1,-1)
        std_auc = f['b'].reshape(1,-1)
        avg_acc = f['c'].reshape(1,-1)
        std_acc = f['d'].reshape(1,-1)
        ALPHAs = f['e'].reshape(1,-1)
        # print(ACCs)
        # print(AUCs)
        if LAMBDA == 0.:
            data = ALPHAs.copy()
        data = np.concatenate((data, avg_acc))
        data = np.concatenate((data, std_acc))
        data = np.concatenate((data, avg_auc))
        data = np.concatenate((data, std_auc))

        plt.figure()
        plt.plot(ALPHAs[0], avg_acc[0], color='darkorange', lw=2, label='accuracy')
        plt.plot(ALPHAs[0], avg_auc[0], color='blue', lw=2, label='AUROC')
        # plt.xlim([0.0, 1.0])
        plt.xlim([0.0, 1.])
        plt.ylim([0.0, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('accuracy, AUROC')
        plt.legend(loc="lower right")
        dir = outputDir + 'acc_auc_lambda{:.2f}.png'.format(LAMBDA)
        plt.savefig(dir)
        print('lambda {:.2f}'.format(LAMBDA), ', max acc {:.4f}'.format(np.amax(avg_acc)))
    data = data.transpose()
    print(data.shape)
    np.savetxt(outputDir+'acc_auc.csv', data, fmt='%.4f')

def plot_auc_acc_duq():

    lambdas = np.linspace(0., 1., 11)
    length_scales = np.linspace(0.1, 1., 10)
    outputDir = '/home/hh/data/ngsim/combined_dataset/duq/'
    aucs = []
    accs = []
    max_acc = 0.
    for k in range(lambdas.shape[0]):
        for j in range(length_scales.shape[0]):
            dir = outputDir + 'mean_std_accs_aucs_lambda{:.3f}_sigma{:.3f}.npz'.format(lambdas[k], length_scales[j])
            f = np.load(dir)
            avg_auc = f['a'].reshape(1,-1)
            std_auc = f['b'].reshape(1,-1)
            avg_acc = f['c'].reshape(1,-1)
            std_acc = f['d'].reshape(1,-1)
            epochs = f['e'].reshape(1,-1)
            aucs.append(avg_auc)
            aucs.append(std_auc)
            accs.append(avg_acc)
            accs.append(std_acc)
            plt.figure()
            plt.plot(epochs[0], avg_acc[0], color='darkorange', lw=2, label='accuracy')
            plt.plot(epochs[0], avg_auc[0], color='blue', lw=2, label='AUROC')
            # plt.xlim([0.0, 1.0])
            plt.xlim([0, 200])
            plt.ylim([0.0, 1.])
            plt.xlabel('$epoch')
            plt.ylabel('accuracy, AUROC')
            plt.legend(loc="lower right")
            dir = outputDir + 'acc_auc_lambda{:.3f}_sigma{:.3f}.png'.format(lambdas[k], length_scales[j])
            plt.savefig(dir)
            maxInd = np.argmax(avg_acc)
            print('lambda{:.3f}_sigma{:.3f}'.format(lambdas[k], length_scales[j]))
            print('acc {:.4f}, {:.4f}'.format(avg_acc[0, maxInd], std_acc[0, maxInd]))
            print('auc {:.4f}, {:.4f}'.format(avg_auc[0, maxInd], std_auc[0, maxInd]))
            if avg_acc[0, maxInd]>max_acc:
                max_acc = avg_acc[0, maxInd]
                max_std_acc = std_acc[0, maxInd]
                max_auc = avg_auc[0, maxInd]
                max_std_auc = std_auc[0, maxInd]

    aucs = np.array(aucs)
    accs = np.array(accs)
    aucs = np.stack(epochs, aucs)
    accs = np.stack((epochs, accs))
    aucs = aucs.transpose()
    accs = accs.transpose()
    np.savetxt(outputDir+'acc.csv', accs, fmt='%.4f')
    np.savetxt(outputDir+'auc.csv', aucs, fmt='%.4f')


def plot_auc_acc_csnn_multiple_r2():
    aucs = []
    accs = []
    r2s = [0.1, 0.5, 1.0, 2.0]
    for r2 in r2s:
        dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r2{:.1f}.npz'.format(r2)
        f = np.load(dir)
        AUCs = f['a']
        ACCs = f['c']
        ALPHAs = f['e']
        aucs.append(AUCs)
        accs.append(ACCs)

    plt.figure()
    for i in range(len(r2s)):
        plt.plot(ALPHAs, accs[i], lw=2, label='$r^2=${:.1f}'.format(r2s[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('ACC')
        plt.legend()
        dir = '/home/hh/data/acc_r2_impact.png'
        plt.savefig(dir)

    plt.figure()
    for i in range(len(r2s)):
        plt.plot(ALPHAs, aucs[i], lw=2, label='$r^2=${:.1f}'.format(r2s[i]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('AUC')
        plt.legend()
        dir = '/home/hh/data/auc_r2_impact.png'
        plt.savefig(dir)
    plt.show()

def plot_layer_effect():
    aucs = []
    accs = []
    outputDir = '/home/hh/data/csnn/'
    labels = ['Net2', 'Net3', 'Net4']
    alphas = None
    for net in labels:
        dir = outputDir + '/mean_std_accs_aucs_'+ net.lower()+'.npz'
        f = np.load(dir)
        AUCs = f['a']
        ACCs = f['c']
        ALPHAs = f['e']
        aucs.append(AUCs)
        accs.append(ACCs)
        if alphas is None: alphas = ALPHAs

    plt.figure()
    for i in range(len(labels)):
        plt.plot(alphas, accs[i], lw=2, label= labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('ACC')
        plt.legend()
    dir = outputDir + 'acc_layers_impact.png'
    # plt.show()
    plt.savefig(dir)

    plt.figure()
    for i in range(len(labels)):
        plt.plot(alphas, aucs[i], lw=2, label=labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('AUC')
        plt.legend()
    dir = outputDir + 'auc_layers_impact.png'
    # plt.show()
    plt.savefig(dir)

    dir = outputDir+'acc_auc_layers_impact.csv'
    aucs = np.array(aucs)
    accs = np.array(accs)
    alphas = alphas.reshape(1, -1)
    print(alphas.shape, aucs.shape, accs.shape)
    data = np.concatenate((alphas, aucs))
    data = np.concatenate((data, accs))
    data = data.transpose()
    rows = [4*i for i in range(10)]
    rows.append(39)
    np.savetxt(dir, data[rows], fmt='%.3e')
    print(data.shape)


def plot_pretrain():
    # dir0 = '/home/hh/data/train_us80_validate_us101/pre_train_acc_loss_'
    dir0 = '/home/hh/data/train_us101_validate_us80/pre_train_acc_loss_'
    nets = ['mlp3', 'mlp4', 'net3', 'net4']
    accs = []
    accs_validate = []
    losses = []
    losses_validate = []
    for net in nets:
        dir = dir0+net+'.npz'
        f = np.load(dir)
        accs.append(f['a'])
        accs_validate.append(f['e'])
        losses.append(f['c'])
        losses_validate.append(f['g'])
    epochs = np.arange(accs[0].shape[0])
    plt.figure()
    for i in range(len(nets)):
        plt.plot(epochs, accs[i], lw=2, label='train_'+nets[i])
        plt.plot(epochs, accs_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 200, 0.5, 1.])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    dir = dir0+'pre_train_acc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)
    print('best validate acc ', np.max(accs_validate, axis=1))

    plt.figure()
    for i in range(len(nets)):
        plt.plot(epochs, losses[i], lw=2, label='train_'+nets[i])
        plt.plot(epochs, losses_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 200, 0., 0.8])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    dir = dir0+'pre_train_loss.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


def plot_train():
    # dir0 = '/home/hh/data/train_us80_validate_us101/mean_std_accs_aucs_'
    dir0 = '/home/hh/data/train_us101_validate_us80/mean_std_accs_aucs_'
    nets = ['net3', 'net4']
    aucs =[]
    accs = []
    accs_validate = []
    losses = []
    losses_validate = []
    alphas =  None
    for net in nets:
        dir = dir0+net+'.npz'
        f = np.load(dir)
        aucs.append(f['a'])
        accs.append(f['c'])
        accs_validate.append(f['e'])
        losses.append(f['g'])
        losses_validate.append(f['i'])
        if alphas is None:
            alphas = f['k']
    epochs = np.arange(accs[0].shape[0])
    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, accs[i], lw=2, label='train_'+nets[i])
        plt.plot(alphas, accs_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 1., 0.5, 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy')
    dir = dir0+'train_acc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)
    print('best validate acc ', np.max(accs_validate, axis=1))

    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, losses[i], lw=2, label='train_'+nets[i])
        plt.plot(alphas, losses_validate[i], lw=2, label='validate_'+nets[i])
    plt.axis([0, 1., 0., 0.8])
    plt.xlabel('$\\alpha$')
    plt.ylabel('loss')
    dir = dir0+'train_loss.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


    plt.figure()
    for i in range(len(nets)):
        plt.plot(alphas, accs_validate[i], lw=2, label='acc_'+nets[i])
        plt.plot(alphas, aucs[i], lw=2, label='auc_' + nets[i])
    for i in range(aucs[0].shape[0]):
        print(accs_validate[0][i], aucs[0][i])
    print('for net 2')
    for i in range(aucs[0].shape[0]):
        print(accs_validate[1][i], aucs[1][i])
    plt.axis([0, 1., 0., 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy, auc')
    dir = dir0 + 'train_acc_auc.png'
    # plt.show()
    plt.legend()
    plt.savefig(dir)


def plot_min_distance_within_dataset():
    dir = '/home/hh/data/ngsim/min_dis_within_us80.npz'
    f = np.load(dir)
    dis_us80 = f['a']
    frequency_us80 = f['b']

    dir = '/home/hh/data/ngsim/min_dis_within_us101.npz'
    f = np.load(dir)
    dis_us101 = f['a']
    frequency_us101 = f['b']

    plt.plot(dis_us80, frequency_us80, label='us80')
    plt.plot(dis_us101, frequency_us101, label='us101')
    plt.axis([0, 3.0, 0., 1.1])
    plt.ylabel('percentage')
    plt.xlabel('minimum distance to in-distribution samples')
    plt.legend()
    plt.show()


def plot_auc_score_functions():
    outputDir = '/home/hh/data/score_function/'
    dir = outputDir + '/mean_std_accs_aucs.npz'
    scores = ['logit', 'softmax', 'energy', 'logit+log_softmax']
    f = np.load(dir)
    aucs = f['a']
    alphas = f['e']
    plt.figure()
    for i in range(len(scores)):
        plt.plot(alphas, aucs[:, i], lw=2, label='auc_' + scores[i])
    plt.axis([0, 1., 0., 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('auc')
    dir = outputDir + 'aucs.png'
    plt.legend()
    plt.show()
    plt.savefig(dir)

def plot_learnable_r():
    # outputDir = '/home/hh/data/moons/radius_penalty_impact/lambda_1.28'
    outputDir = '/home/hh/data/two_gaussian/radius_penalty_impact/lambda_1.28'
    dir = outputDir + '/mean_std_accs_aucs_net4.npz'
    f = np.load(dir)
    rs = f['l']
    print(rs.shape)
    epochs = (np.arange(rs.shape[0])+1)*5
    plt.figure()
    plt.plot(epochs, rs[:,0], label='$||r||_\infty$')
    plt.plot(epochs, rs[:,1]/8., label='$||r||_2/\sqrt{n_{hidden}}$')
    plt.axis([0, 500, 0., 1.8])
    plt.ylabel('r norm')
    plt.xlabel('epochs')
    plt.legend()
    # plt.show()
    dir = outputDir + '/learnable_r.png'
    plt.savefig(dir)


def plot_circles(layer, x_train, y_train, alpha, r, epoch, dir0, bias=False):
    figure, ax = plt.subplots()
    mask = y_train.astype(np.bool)
    w = layer.weight.data.numpy()
    # w is of shape (64,2)
    center = w/alpha
    radius2 = r*r + np.sum(w*w, axis=1)*(1/alpha/alpha-1)
    if bias:
        b = layer.bias.data.numpy()
        # b is of shape (64,)
        radius2 -= (1.-b/alpha)*(1.-b/alpha)
        radius2[radius2<0.] = 0.
    radius = np.sqrt(radius2)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    N = w.shape[0]
    col = 1. / N * np.arange(0, N)
    for i in range(w.shape[0]):
        # circle = plt.Circle(center[i], radius[i], fill=False)
        # circle = plt.Circle(center[i], radius[i], color=str(col[i]))
        circle = plt.Circle(center[i], radius[i], color=cm.jet(col[i]), fill=False)
        ax.add_artist(circle)
        ax.axis('equal')
        # ax.autoscale(False)

    plt.scatter(x_train[mask, 0], x_train[mask, 1], zorder=1)
    plt.scatter(x_train[~mask, 0], x_train[~mask, 1], zorder=1)
    dir = dir0 + '/confidence_circle_epoch_{}.png'.format(epoch)
    # mean = [0.50048237, 0.24869538]
    # std = [0.8702328,  0.50586896]
    # x_range = [-2.5, 3.5]
    # y_range = [-3., 3.]
    ax.set_xlim([-3.,3.])
    ax.set_ylim([-3.,3.])
    # plt.axis([-3, 3., -3, 3])
    plt.savefig(dir)

def plot_two_gaussian(x_train):
    figure, ax = plt.subplots()
    mask = x_train[:, -1] == 0
    plt.scatter(x_train[mask, 0], x_train[mask, 1], zorder=1)
    plt.scatter(x_train[~mask, 0], x_train[~mask, 1], zorder=1)
    ax.set_xlim([-8.,8.])
    ax.set_ylim([-8.,8.])
    # plt.show()
    # plt.axis([-3, 3., -3, 3])
    dir = '/home/hh/data/two_gaussian/'
    plt.savefig(dir+'two_gaussian.png')

def plot_radius_penalty_impact_on_acc():
    dir0 = '/home/hh/data/two_gaussian/radius_penalty_impact/'
    lambdas = [0.0, 0.02, 0.08, 0.16, 0.32, 0.64, 1.28]
    accs = []
    accs_valid = []
    for l in lambdas:
        dir = dir0+'lambda_{:.2f}'.format(l)
        dir = dir + '/mean_std_accs_aucs_net4.npz'
        data = np.load(dir)
        acc = np.mean(data['c'][-10:])
        acc_valid = np.mean(data['e'][-10:])
        accs.append(acc)
        accs_valid.append(acc_valid)
    figure, ax = plt.subplots()
    plt.plot(lambdas, accs, label='train')
    plt.plot(lambdas, accs_valid, label='test')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('$\lambda$')
    # plt.show()
    plt.savefig(dir0+'radius_penalty_on_accuracy.png')

def plot_confidence_map_alpha_impact():
    dir0 = '/home/hh/data/moons/alpha_impact/alpha_0/'
    dir = dir0 + 'moons_confidence_alpha0.npz'
    data = np.load(dir)
    x_lin = data['a']
    y_lin = data['b']
    z0 = data['c']
    X_vis = data['d']
    mask = data['e']
    dir = dir0 + 'moons_confidence_alpha1.npz'
    data = np.load(dir)
    z1 = data['c']

    dir = '/home/hh/data/uncertainty_deep_ensemble_moons'
    np.savez(dir + ".npz", a=x_lin, b=y_lin, z=confidence)

    z = [z0, z1]
    axs = []
    fig = plt.figure()
    l = np.linspace(0.5, 1., 21)
    titles = ['$\\alpha=0$', '$\\alpha=1$']
    for i in range(len(z)):
        axs.append(fig.add_subplot(1, 2, i+1))
        plt.contourf(x_lin, y_lin, z[i], cmap=plt.get_cmap('inferno'), levels=l)  # , extend='both')
        if(i==1):
            plt.colorbar()
        plt.scatter(X_vis[mask, 0], X_vis[mask, 1])
        plt.scatter(X_vis[~mask, 0], X_vis[~mask, 1])
        axs[i].set_title(titles[i])
    # fig.tight_layout()
    # plt.axis([-3, 3., -3, 3])
    # dir0 = '/home/hh/data/moons/'
    dir = dir0 + 'confidence_alpha_impact.png'
    plt.savefig(dir)
    # plt.show()


if __name__ == "__main__":
    plot_auc_acc_csnn()
    # plot_auc_acc_duq()
    # plot_min_distance_within_dataset()
    # plot_pretrain()
    # plot_train()
    # plot_auc_score_functions()
    # plot_layer_effect()
    # plot_learnable_r()
    # plot_radius_penalty_impact_on_acc()
    # plot_confidence_map_alpha_impact()
    # plot_save_acc_average_std()
