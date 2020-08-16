###########################################################################
   # loss function changed to SVM
   # 3.21.2020 now instead of majority vote, stack snapshots as a big
   # vector to do prediction
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

seqLen=3
features=10
inputs=features#*seqLen
seed0=1013899
seed1=1018859
units=96
#units=64
learningRate=0.0001

torch.manual_seed(seed0)

#delta=seqLen/2.
#delta=seqLen/8.
#C=0.1
dropoutRate=0.5

l2Penalty=5.0e-3

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
    sz=int(x_validate.shape[0])
    y_predict=-np.ones(sz)
    loss=0
    for i in range(sz):
        output=outputs[i]

        labels=y_validate[i]
        lossRaw=1.-labels*output
        lossRaw[lossRaw<0.]=0.
        loss+=lossRaw

        y=output
        if y>0.: 
            y_predict[i]=1.
    loss/=sz
    temp=y_predict==y_validate
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

#def analysis(x_test0, y_predict, y_label, w0, predict0):
def analysis(x_test, y_predict, y_label):
    #visualize_prediction(x_test0, y_test, y_predict)
    x_test0=x_test[::seqLen]
    #for merge after
    label=y_label[x_test0[:,0]==1]
    predict=y_predict[x_test0[:,0]==1]
    accuracy=np.mean(label==predict)
    print('merge after', label.shape[0],', accuracy for merge after samples', accuracy)
    #for merge infront
    label=y_label[x_test0[:,0]==0]
    predict=y_predict[x_test0[:,0]==0]
    accuracy=np.mean(label==predict)
    print('merge infront', label.shape[0], ', accuracy for merge infront samples', accuracy)

    label=y_label
    predict=y_predict
    #print('check ', np.mean(predict0==predict))
    accuracy=np.mean(predict==label)
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
    #dirs="data_begin_with_lateral_move_3_data_point_per_sample/"
    #f=np.load(dirs+'train_normalized.npz')
    f=np.load('train_normalized.npz')
    x_train=f['a']
    y_train=f['b']
    y_train[y_train==0]=-1
    if testSizeEffect:
        sz=int(x_train.shape[0]/2)
        x_train=x_train[:sz]
        y_train=y_train[:sz]
    print('x_train', x_train.shape[0], np.mean(y_train==1),'coop rate')

    #f=np.load(dirs+'validate_normalized.npz')
    f=np.load('validate_normalized.npz')
    x_validate=f['a']
    y_validate=f['b']
    y_validate[y_validate==0]=-1
    if testSizeEffect:
        sz=int(x_validate.shape[0]/2)
        x_validate=x_validate[:sz]
        y_validate=y_validate[:sz]
    print('x_validate', x_validate.shape[0], np.mean(y_validate==1),'coop rate')
    
    #f=np.load('test_normalized.npz')
    #x_test=f['a']
    #y_test=f['b']
    #y_test[y_test==0]=-1
    #
    #if testSizeEffect:
    #    sz=int(x_test.shape[0]/2)
    #    x_test=x_test[:sz]
    #    y_test=y_test[:sz]
    #print('x_test', x_test.shape[0], np.mean(y_test==1),'coop rate')
    
    #f=np.load(dirs+'train_origin.npz')
    f=np.load('train_origin.npz')
    x_train0=f['a']
    if testSizeEffect:
        sz=int(x_train0.shape[0]/2)
        x_train0=x_train0[:sz]
    print(x_train0.shape)

    #f=np.load(dirs+'validate_origin.npz')
    f=np.load('validate_origin.npz')
    x_validate0=f['a']
    if testSizeEffect:
        sz=int(x_validate0.shape[0]/2)
        x_validate0=x_validate0[:sz]
    print(x_validate0.shape)

    #f=np.load('test_origin.npz')
    #x_test0=f['a']
    #if testSizeEffect:
    #    sz=int(x_test0.shape[0]/2)
    #    x_test0=x_test0[:sz]
    #print(x_test0.shape)
    #return (x_train0, x_train, y_train, x_validate0, x_validate,
    #        y_validate, x_test0, x_test, y_test)
    return (x_train0, x_train, y_train, x_validate0, x_validate, y_validate)

(x_train0, x_train, y_train, x_validate0, x_validate, y_validate)=loadData()

x_train=x_train[:,20:]
x_validate=x_validate[:,20:]

batchSize=1
dataset_train = TensorDataset(Tensor(x_train), Tensor(y_train) )
train_loader = torch.utils.data.DataLoader(dataset_train, 
                     batch_size=batchSize,shuffle=False)

dataset_validate = TensorDataset(Tensor(x_validate), Tensor(y_validate) )
validate_loader = torch.utils.data.DataLoader(dataset_validate, 
                     batch_size=batchSize,shuffle=False)

#dataset_test = TensorDataset(Tensor(x_test), Tensor(y_test) )
#test_loader = torch.utils.data.DataLoader(dataset_test, 
#                     batch_size=batchSize,shuffle=False)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=learningRate,
                       weight_decay=l2Penalty)
# calculate weights in loss function
#w0,w1=cal_weights()
#
#w0=torch.zeros_like(w0)
#w0[-1]=1.*seqLen
#w1=torch.zeros_like(w1)
#w1[-1]=1.*seqLen

# train network
losses=[]
accuracies=[]
losses_validate=[]
accuracies_validate=[]

PATH = './cifar_net.pth'

trained=False
#trained=True
if not trained:
    for epoch in range(150):  # loop over the dataset multiple times
        running_loss = 0.0
        misClassify=0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if(i==0 and epoch==0):
                print('inputs, labels shape', inputs.shape, labels.shape )
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            #print('label', labels)
            #print('outputs', outputs)
            outputs = outputs[:,0]
            loss=1-labels*outputs
            loss[loss<0.]=0.
            #print(lossRaw)
            #loss=(lossRaw*w0).sum()/seqLen
            loss.backward()
            optimizer.step()
            
            y_predict=outputs
            if y_predict>0.: 
                y_predict=1.
            else:
                y_predict=-1.

            if y_predict!=labels[0].item():
                misClassify+=1
    
            # print statistics
            running_loss += loss.item()
            #if i % 50 == 49:    # print every 50 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch + 1, i + 1, running_loss / 50))
            #    running_loss = 0.0
        accuracy=1-misClassify/x_train.shape[0]
        loss=running_loss/x_train.shape[0]
        losses.append(loss)
        accuracies.append(accuracy)
        print('epoch', epoch, ', loss', loss, 'accuracy', accuracy)
    
        loss, accuracy=infer(net, x_validate, y_validate)
        net.train() # switch back to train model, in which dropout is turned on
        print('validate: epoch', epoch, ', loss', loss, 'accuracy', accuracy)
        losses_validate.append(loss)
        accuracies_validate.append(accuracy)
        #if epoch==117:
        #if epoch==63:
        if epoch==113:
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

print('for train')
net.eval() # turn on eval, switch off dropout layer
with torch.no_grad():
    y_predict=net(Tensor(x_train))
y_predict=y_predict[:,0]
predict=torch.ones_like(y_predict)
predict[y_predict<0]=-1
analysis(x_train0, predict.numpy(), y_train)
y_train[y_train==-1]=0.
predict[predict==-1]=0.
vize.visualize_sample(x_train0[seqLen-1::seqLen], y_train)
vize.visualize_prediction(x_train0[seqLen-1::seqLen], y_train, predict.numpy())

print('for validate')
net.eval() # turn on eval, switch off dropout layer
with torch.no_grad():
    y_predict=net(Tensor(x_validate))
y_predict=y_predict[:,0]
predict=torch.ones_like(y_predict)
predict[y_predict<0]=-1
analysis(x_validate0, predict.numpy(), y_validate)
y_validate[y_validate==-1]=0.
predict[predict==-1]=0.
vize.visualize_sample(x_validate0[seqLen-1::seqLen], y_validate)
vize.visualize_prediction(x_validate0[seqLen-1::seqLen], y_validate, predict.numpy())
#
#print('for test')
#with torch.no_grad():
#    y_predict=net(Tensor(x_test))
#y_predict=y_predict[:,0]
#predict=torch.ones_like(y_predict)
#predict[y_predict<0]=-1
#analysis(x_test0, predict.numpy(), y_test)
#y_test[y_test==-1]=0.
#predict[predict==-1]=0.
#vize.visualize_sample(x_test0[seqLen-1::seqLen], y_test)
#vize.visualize_prediction(x_test0[seqLen-1::seqLen], y_test,predict.numpy())
