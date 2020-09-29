import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

def plot_acc():
    acc = np.loadtxt('/home/hh/data/lambda0.txt')
    plt.figure()
    plt.plot(acc[:,0], acc[:,1], '-o')
    plt.axis([0, 1.0, 0.8, 0.85])
    plt.ylabel('average best validation acc')
    plt.xlabel('$\sigma $')
    plt.show()

def plot_save_loss(losses, losses_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(losses)) + 1, losses, label='train')
    plt.plot(np.arange(len(losses)) + 1, losses_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.axis([0, 200, 0.2, 1.8])
    plt.legend()
    plt.savefig(dir)


def plot_save_acc(accuracies, accuracies_validate, dir):
    plt.figure()
    plt.plot(np.arange(len(accuracies)) + 1, accuracies, label='train')
    plt.plot(np.arange(len(accuracies)) + 1, accuracies_validate, label='validate')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0, 200, 0.6, 0.85])
    plt.legend()
    plt.savefig(dir)


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
    plt.figure()

    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers.npz'
    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_trim.npz'
    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r22.0.npz'
    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r20.1.npz'
    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r22.0_maxAlpha0.2.npz'
    # dir = '/home/hh/data/mean_std_accs_aucs_csnn_4_layers_batchNorm.npz'
    dir = '/home/hh/data/mean_std_accs_aucs_csnn_maxAlpha1.00_4_layers_batchNorm.npz'
    f = np.load(dir)
    AUCs = f['a']
    ACCs = f['c']
    ALPHAs = f['e']
    # print(ACCs)
    # print(AUCs)
    outputs=[]
    for i in range(AUCs.shape[0]):
        outputs.append([ACCs[i], AUCs[i]])
    outputs = np.array(outputs)
    # outputs = np.array(outputs).reshape(-1,2)
    # print(outputs)
    # np.savetxt('/home/hh/data/acc_auc_csnn_4_layers_batchNorm.csv', outputs, fmt='%.3f')
    np.savetxt('/home/hh/data/acc_auc_csnn_4_layers_batchNorm_maxAlpha1.00.csv', outputs, fmt='%.3f')
    plt.plot(ALPHAs, ACCs, color='darkorange', lw=2, label='accuracy')
    plt.plot(ALPHAs, AUCs, color='blue', lw=2, label='auc')
    plt.xlim([0.0, 1.0])
    # plt.xlim([0.0, .05])
    plt.ylim([0.0, 1.])
    plt.xlabel('$\\alpha$')
    plt.ylabel('accuracy, AUC')
    plt.legend(loc="lower right")
    # dir = '/home/hh/data/acc_auc_csnn_2_layers.png'
    # dir = '/home/hh/data/acc_auc_csnn_2_layers_trim.png'
    # dir = '/home/hh/data/acc_auc_csnn_2_layers_r22.0.png'
    # dir = '/home/hh/data/acc_auc_csnn_2_layers_r20.1.png'
    # dir = '/home/hh/data/acc_auc_csnn_2_layers_r22.0_maxAlpha0.2.png'
    dir = '/home/hh/data/acc_auc_csnn_4_layers_batchNorm_maxAlpha1.00.png'
    plt.savefig(dir)
    # plt.show()

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

    r2 = 1.0
    dir = '/home/hh/data/mean_std_accs_aucs_csnn_2_layers_r2{:.1f}.npz'.format(r2)
    f = np.load(dir)
    AUCs = f['a']
    ACCs = f['c']
    ALPHAs = f['e']
    aucs.append(AUCs)
    accs.append(ACCs)

    dir = '/home/hh/data/mean_std_accs_aucs_csnn_4_layers.npz'
    f = np.load(dir)
    AUCs = f['a']
    ACCs = f['c']
    # ALPHAs = f['e']
    aucs.append(AUCs)
    accs.append(ACCs)

    plt.figure()
    labels = ['2 layers', '4 layers']
    for i in range(2):
        plt.plot(ALPHAs, accs[i], lw=2, label= labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.5, 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('ACC')
        plt.legend()
    dir = '/home/hh/data/acc_layers_impact.png'
    # plt.show()
    plt.savefig(dir)

    plt.figure()
    for i in range(2):
        plt.plot(ALPHAs, aucs[i], lw=2, label=labels[i])
        plt.xlim([0.0, 1.0])
        plt.ylim([0., 1.])
        plt.xlabel('$\\alpha$')
        plt.ylabel('AUC')
        plt.legend()
    dir = '/home/hh/data/auc_r2_impact.png'
    # plt.show()
    plt.savefig(dir)

# plot_layer_effect()
# plot_feature_selections()
# plot_acc()
# plot_auc()
# plot_func()
plot_auc_acc_csnn()
# plot_auc_acc_csnn_multiple_r2()
