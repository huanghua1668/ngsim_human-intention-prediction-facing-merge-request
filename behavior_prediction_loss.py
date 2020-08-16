import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import matplotlib.pyplot as plt

inputs=10
seed0=1013899
seed1=1018859
units=64
learningRate=0.0002
hiddenLayers=3

torch.manual_seed(seed0)

seqLen=10
#delta=seqLen/2.
delta=seqLen/8.
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
        #self.drop1=nn.Dropout(p=0.5, inplace=False)
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
    with torch.no_grad():
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

def calculate_label(predict0, w0):
    sz=int(predict0.shape[0]/seqLen)
    predict=np.ones(sz)
    for i in range(sz):
        y=(((w0*predict0[i*seqLen:i*seqLen+seqLen]).sum())/seqLen).item()>0.5
        #y=(((w0*predict0[i*seqLen:i*seqLen+seqLen]).sum())/seqLen).item()>0.6
        predict[i]=int(y)
    return predict

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

testSizeEffect=False
# load data
f=np.load('train_normalized.npz')
x_train=f['a']
y_train=f['b']

if testSizeEffect:
    sz=int(x_train.shape[0]/2)
    x_train=x_train[:sz]
    y_train=y_train[:sz]

print('x_train', x_train.shape[0]/seqLen, y_train.mean(),'coop rate')
f=np.load('validate_normalized.npz')
x_validate=f['a']
y_validate=f['b']

if testSizeEffect:
    sz=int(x_validate.shape[0]/2)
    x_validate=x_validate[:sz]
    y_validate=y_validate[:sz]

print('x_validate', x_validate.shape[0]/seqLen, y_validate.mean(),'coop rate')
f=np.load('test_normalized.npz')
x_test=f['a']
y_test=f['b']

if testSizeEffect:
    sz=int(x_test.shape[0]/2)
    x_test=x_test[:sz]
    y_test=y_test[:sz]

print('x_test', x_test.shape[0]/seqLen, y_test.mean(),'coop rate')

f=np.load('validate_origin.npz')
x_validate0=f['a']
if testSizeEffect:
    sz=int(x_validate0.shape[0]/2)
    x_validate0=x_validate0[:sz]
print(x_validate0.shape)
f=np.load('test_origin.npz')
x_test0=f['a']
if testSizeEffect:
    sz=int(x_test0.shape[0]/2)
    x_test0=x_test0[:sz]
print(x_test0.shape)

dataset_train = TensorDataset(Tensor(x_train), Tensor(y_train) )
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=seqLen,shuffle=False)

dataset_validate = TensorDataset(Tensor(x_validate), Tensor(y_validate) )
validate_loader = torch.utils.data.DataLoader(dataset_validate, batch_size=seqLen,shuffle=False)

dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test) )
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=seqLen,shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=learningRate)

# calculate weights in loss function
w0,w1=cal_weights()

#w0=torch.zeros_like(w0)
#w0[0]=1.*seqLen
#w1=torch.zeros_like(w1)
#w1[0]=1.*seqLen

# train network
losses=[]
accuracies=[]
losses_validate=[]
accuracies_validate=[]

PATH = './cifar_net.pth'

trained=False
#trained=True
if not trained:
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
        if epoch==15:
            torch.save(net.state_dict(), PATH)
    
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
    trained=True

net.load_state_dict(torch.load(PATH))
print('load parameters successfully')

print('for validate')
with torch.no_grad():
    y_predict=net(Tensor(x_validate))
y_predict=y_predict[:,0].numpy()
analysis(x_validate0, y_predict, y_validate, w0)
print('for test')
with torch.no_grad():
    y_predict=net(Tensor(x_test))
y_predict=y_predict[:,0].numpy()
analysis(x_test0, y_predict, y_test, w0)
