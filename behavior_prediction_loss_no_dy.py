import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt

inputs=7
seed0=1013899
seed1=1018859
units=32
learningRate=0.0002
hiddenLayers=3

torch.manual_seed(seed0)

seqLen=10
delta=seqLen/2.
#delta=seqLen/1.
C=0.1

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
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, 1)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        p = torch.sigmoid(x)
        return p

def infer(net, x_validate, y_validate):
    x_validate=Tensor(x_validate)
    outputs = net(x_validate)
    outputs = outputs[:,0]
    sz=int(x_validate.shape[0]/seqLen)
    y_predict=np.zeros(sz)
    loss=0
    for i in range(sz):
        output=outputs[i*seqLen:(i*seqLen+seqLen)]

        labels=Tensor(y_validate[i*seqLen:(i*seqLen+seqLen)])
        lossRaw=-(labels*torch.log(output)+(1.-labels)*torch.log(1.-output))
        loss0=(lossRaw*w0).sum()/seqLen
        u=output>0.5
        u=(u.numpy()).astype(int)
        u=torch.tensor(u)
        lossRaw=u[1:]-u[:-1]
        loss1=(w1*lossRaw*lossRaw).sum()/(seqLen-1)
        loss+=loss0+loss1*C

        y=(((w0*output).sum())/seqLen).item()
        if y>0.5: 
            y_predict[i]=1.
    loss/=sz
    temp=y_predict==y_validate[::seqLen]
    accuracy=(temp.astype(int)).mean()
    return loss, accuracy

# load data
f=np.load('train_normalized.npz')
x_train=f['a']
x_train=x_train[:,:-3]
print('x_train shape', x_train.shape)
y_train=f['b']
f=np.load('validate_normalized.npz')
x_validate=f['a']
x_validate=x_validate[:,:-3]
y_validate=f['b']
f=np.load('test_normalized.npz')
x_test=f['a']
x_test=x_test[:,:-3]
y_test=f['b']

dataset_train = TensorDataset(Tensor(x_train), Tensor(y_train) )
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=10,shuffle=False)

dataset_validate = TensorDataset(Tensor(x_validate), Tensor(y_validate) )
validate_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=10,shuffle=False)

dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test) )
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=10,shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=learningRate)

# calculate weights in loss function
w0,w1=cal_weights()

# train network
losses=[]
accuracies=[]
losses_validate=[]
accuracies_validate=[]
for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    misClassify=0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if(i==0 and epoch==0):
            print('inputs', inputs)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print('label', labels)
        #print('outputs', outputs)
        outputs = outputs[:,0]
        lossRaw=-(labels*torch.log(outputs)+(1.-labels)*torch.log(1.-outputs))
        #print(lossRaw)
        loss0=(lossRaw*w0).sum()/seqLen
        u=outputs>0.5
        u=(u.numpy()).astype(int)
        u=torch.tensor(u)
        lossRaw=u[1:]-u[:-1]
        loss1=(w1*lossRaw*lossRaw).sum()/(seqLen-1)
        #print(i, loss0, loss1)
        loss=loss0+loss1*C
        loss.backward()
        optimizer.step()
        
        y_predict=(((w0*outputs).sum())/seqLen).item()
        if y_predict>0.5: 
            y_predict=1.
        else:
            y_predict=0.
        if y_predict!=labels[0].item():
            misClassify+=1

        # print statistics
        running_loss += loss.item()
        #if i % 50 == 49:    # print every 50 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 50))
        #    running_loss = 0.0
    accuracy=1-misClassify/x_train.shape[0]*seqLen
    loss=running_loss/x_train.shape[0]*seqLen
    losses.append(loss)
    accuracies.append(accuracy)
    print('epoch', epoch, ', loss', loss, 'accuracy', accuracy)

    loss, accuracy=infer(net, x_validate, y_validate)
    print('validate: epoch', epoch, ', loss', loss, 'accuracy', accuracy)
    losses_validate.append(loss)
    accuracies_validate.append(accuracy)

plt.figure()
plt.plot(np.arange(len(losses))+1, losses, label='train')
plt.plot(np.arange(len(losses))+1, losses_validate, label='validate')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.figure()
plt.plot(np.arange(len(losses))+1, accuracies, label='train')
plt.plot(np.arange(len(losses))+1, accuracies_validate, label='validate')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

print('Finished Training')
