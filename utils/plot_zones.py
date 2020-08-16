###########################################################################
   # loss function changed to SVM
###########################################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt

import visualize as vize

inputs=10
seed0=1013899
seed1=1018859
units=64
#units=96
learningRate=0.0001
hiddenLayers=3

torch.manual_seed(seed0)

seqLen=10
#delta=seqLen/2.
delta=seqLen/8.
C=0.1
dropoutRate=0.3
l2Penalty=1.0e-3

# calculate weights for loss function
def cal_weights():
    t=np.arange(seqLen)
    t=(t-(seqLen-1))/delta
    weights=np.exp(-t*t)
    sum0=weights.sum()
    sum1=weights[1:].sum()
    w0=weights/sum0*seqLen
    w1=weights[1:]/sum1*(seqLen-1)
    print('w0', w0)
    print('w1', w1)
    return Tensor(w0),Tensor(w1)

# build a MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(inputs, units)
        self.fc1 = nn.Linear(units, units)
        #self.drop1=nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, 1)
        self.drop = nn.Dropout(p=dropoutRate)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x=self.drop(x)
        x = F.relu(self.fc1(x))
        x=self.drop(x)
        x = F.relu(self.fc2(x))
        x=self.drop(x)
        x = self.fc3(x)
        #p = torch.sigmoid(x)
        return x

def infer(net, x_validate, y_validate):
    x_validate=Tensor(x_validate)
    net.eval() # turn off dropout layer
    with torch.no_grad():
        outputs = net(x_validate)
    outputs = outputs[:,0]
    sz=int(x_validate.shape[0]/seqLen)
    y_predict=-np.ones(sz)
    loss=0
    for i in range(sz):
        output=outputs[i*seqLen:(i*seqLen+seqLen)]

        labels=Tensor(y_validate[i*seqLen:(i*seqLen+seqLen)])
        lossRaw=1.-labels*output
        lossRaw[lossRaw<0.]=0.
        loss+=(lossRaw*w0).sum()/seqLen

        y=(((w0*output).sum())/seqLen).item()
        if y>0.: 
            y_predict[i]=1.
    loss/=sz
    temp=y_predict==y_validate[::seqLen]
    accuracy=(temp.astype(int)).mean()
    return loss, accuracy

def calculate_label(predict0, w0):
    sz=int(predict0.shape[0]/seqLen)
    predict=-np.ones(sz)
    for i in range(sz):
        y=(((w0*predict0[i*seqLen:i*seqLen+seqLen]).sum())/seqLen).item()>0.
        #y=(((w0*predict0[i*seqLen:i*seqLen+seqLen]).sum())/seqLen).item()>0.6
        if y:
            predict[i]=1.
    return predict

def calculate_probalbility(predict0, w0):
    sz=int(predict0.shape[0]/seqLen)
    predict=-np.ones(sz)
    for i in range(sz):
        y=(((w0*predict0[i*seqLen:i*seqLen+seqLen]).sum())/seqLen).item()
        predict[i]=y
    return predict

#def analysis(x_test0, y_predict, y_label, w0, predict0):
def analysis(x_test0, y_predict, y_label, w0):
    #visualize_prediction(x_test0, y_test, y_predict)
    
    #for merge after
    label=y_label[x_test0[:,0]==1]
    label=label[::seqLen]
    predict=y_predict[x_test0[:,0]==1]
    predict=calculate_label(predict, w0)
    accuracy=np.mean(label==predict)
    print('merge after', label.shape[0],', accuracy for merge after samples', accuracy)
    #for merge infront
    label=y_label[x_test0[:,0]==0]
    label=label[::seqLen]
    predict=y_predict[x_test0[:,0]==0]
    predict=calculate_label(predict, w0)
    accuracy=np.sum(label==predict)/label.shape[0]
    print('merge infront', label.shape[0], ', accuracy for merge infront samples', accuracy)

    label=y_label[::seqLen]
    predict=calculate_label(y_predict, w0)
    #print('check ', np.mean(predict0==predict))
    accuracy=np.sum(predict==label)/predict.shape[0]
    print('overall accuracy', accuracy)
    truePos=np.sum(label)
    trueNeg=label.shape[0]-np.sum(label)
    predictedPos=np.sum(predict)
    predictedNeg=predict.shape[0]-np.sum(predict)
    print('true positive ', truePos)
    print('true negative ', trueNeg)
    print('predicted positive ', predictedPos)
    print('predicted negative ', predictedNeg)

def loadData():
    testSizeEffect=False
    # load data
    f=np.load('train_normalized.npz')
    x_train=f['a']
    y_train=f['b']
    y_train[y_train==0]=-1
    
    print('x_train', x_train.shape[0]/seqLen, y_train.mean(),'coop rate')
    f=np.load('validate_normalized.npz')
    x_validate=f['a']
    y_validate=f['b']
    y_validate[y_validate==0]=-1
    
    print('x_validate', x_validate.shape[0]/seqLen, y_validate.mean(),'coop rate')
    f=np.load('test_normalized.npz')
    x_test=f['a']
    y_test=f['b']
    y_test[y_test==0]=-1
    
    print('x_test', x_test.shape[0]/seqLen, y_test.mean(),'coop rate')
    
    f=np.load('train_origin.npz')

    x_train0=f['a']
    f=np.load('validate_origin.npz')
    x_validate0=f['a']

    f=np.load('test_origin.npz')
    x_test0=f['a']
    return (x_train0, x_train, y_train, x_validate0, x_validate,
            y_validate, x_test0, x_test, y_test)

def loadData_both_dataset():
    # load data
    f=np.load('train_normalized_us80.npz')
    x_train=f['a']
    y_train=f['b']
    y_train[y_train==0]=-1
    f=np.load('validate_normalized_us101.npz')
    x_train1=f['a']
    y_train1=f['b']
    y_train1[y_train1==0]=-1
    x_train=np.concatenate((x_train, x_train1))
    y_train=np.concatenate((y_train, y_train1))
    
    print('x_train', x_train.shape[0]/seqLen, y_train.mean(),'coop rate')
    
    f=np.load('train_origin_us80.npz')
    x_train0=f['a']

    f=np.load('validate_origin_us101.npz')
    x_validate0=f['a']
    x_train0=np.concatenate((x_train0, x_validate0))

    return (x_train0, x_train, y_train)


loadUS80=False
if loadUS80:
    (x_train0, x_train, y_train, x_validate0, x_validate,
     y_validate, x_test0, x_test, y_test)=loadData()
    
    
    x_train0=np.concatenate((x_train0, x_validate0))
    x_train0=np.concatenate((x_train0, x_test0))
    x_train=np.concatenate((x_train, x_validate))
    x_train=np.concatenate((x_train, x_test))
    print(y_train.shape)
    y_train=np.concatenate((y_train, y_validate))
    y_train=np.concatenate((y_train, y_test))

loadBoth=True
if loadBoth:
    (x_train0, x_train, y_train)=loadData_both_dataset()

dataset_train = TensorDataset(Tensor(x_train), Tensor(y_train) )
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=seqLen,shuffle=False)

#dataset_validate = TensorDataset(Tensor(x_validate), Tensor(y_validate) )
#validate_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=seqLen,shuffle=False)
#
#dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test) )
#test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=seqLen,shuffle=False)

net = Net()
# calculate weights in loss function
w0,w1=cal_weights()

#w0=torch.zeros_like(w0)
#w0[0]=1.*seqLen
#w1=torch.zeros_like(w1)
#w1[0]=1.*seqLen

PATH = './cifar_net.pth'

net.load_state_dict(torch.load(PATH))
print('load parameters successfully')

print('for train')
net.eval() # turn on eval, switch off dropout layer
with torch.no_grad():
    y_predict=net(Tensor(x_train))
predict=calculate_label(y_predict[:,0].numpy(), w0)
predict_prob=calculate_probalbility(y_predict[:,0].numpy(), w0)
analysis(x_train0, y_predict[:,0].numpy(), y_train, w0)
y_train[y_train==-1]=0.
predict[predict==-1]=0.
vize.visualize_sample(x_train0[::seqLen], y_train[::seqLen])
vize.visualize_prediction(x_train0[::seqLen], y_train[::seqLen], predict)

x_train0=x_train0[::seqLen]
y_train=y_train[::seqLen]
y_train[y_train==0]=-1
#predict[predict==0]=-1
data=x_train0[:,1:]
#fig, ax=plt.subplots()

m=1
n=1

vMin=-15
vMax=15
deltaV=1.
dv=deltaV/m
#deltaV=5
xMin=-10
xMax=100
deltaX=5
dx=deltaX/n
#deltaX=10
nv=int((vMax-vMin)/dv)+1
nx=int((xMax-xMin)/dx)+1
scores=np.zeros((nv, nx, 2))

dvInd=3
dxInd=6
for k in range(data.shape[0]):
    v=data[k, dvInd]
    x=data[k, dxInd]
    i0=int((v-vMin-deltaV)/dv)+1
    i1=int((v-vMin+deltaV)/dv)
    j0=int((x-xMin-deltaX)/dx)+1
    j1=int((x-xMin+deltaX)/dx)
    for i in range(i0, i1+1):
        for j in range(j0, j1+1):
            if i>=0 and i<nv and j>=0 and j<nx:
                scores[i, j, 0]+=predict[k]
                #scores[vInd, xInd, 0]+=predict_prob[k]
                scores[i, j, 1]+=1
    
for i in range(nv):
    for j in range(nx):
        #if scores[i,j,1]>0:
        if scores[i,j,1]>3:
            scores[i, j, 0]/=scores[i,j,1]
        else:
            #scores[i, j, 0]=-2.
            scores[i, j, 0]=np.nan

for i in range(nv):
    for j in range(nx):
        print(i,j,scores[i,j,0],scores[i,j,1])
    print(scores[i,:,0])

x=np.arange(vMin, vMax+dv, dv) 
y=np.arange(xMin, xMax+dx, dx)
x,y=np.meshgrid(x,y)

fig, ax=plt.subplots()
levels=[0., 0.25, 0.5, 0.75, 1.]
z=np.transpose(scores[:,:,0])
#cs=ax.contourf(x,y,np.transpose(scores[:,:,0]), levels)
#cs=ax.contourf(x,y,z, np.linspace(z.min(),z.max(),10))
#cs=ax.contourf(x,y,z, np.linspace(-1,1,40))
cs=ax.contour(x,y,z, np.linspace(0,1,11))
#cs=ax.contourf(x,y,z, levels)
#cs=ax.contourf(x,y,z)
ax.clabel(cs, cs.levels, inline=False, fontsize=10)

plt.ylabel('$\Delta x$')
plt.xlabel('$\Delta v$')
plt.axis([-10, 5, -10, 50])
#plt.axis([-7, 2, 0, 50])
#ax.legend()
plt.show()


