import numpy as np
import tensorflow as tf
#from tensorflow import keras
import  keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM

inputs=10
seed0=1013899
seed1=1018859
seed2=1002527
seeds=[seed0, seed1, seed2]
units=32
learningRate=0.0002
hiddenLayers=3
dropoutRate=0.3
l2Penalty=1.0e-3

seqLen=10

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

def loadData():
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
    
    f=np.load('train_origin.npz')
    x_train0=f['a']
    if testSizeEffect:
        sz=int(x_train0.shape[0]/2)
        x_train0=x_train0[:sz]
    print(x_train0.shape)

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
    return (x_train0, x_train, y_train, x_validate0, x_validate,
            y_validate, x_test0, x_test, y_test)

(x_train0, x_train, y_train, x_validate0, x_validate,
 y_validate, x_test0, x_test, y_test)=loadData()

x_train=x_train.reshape(-1, seqLen, inputs)
x_validate=x_validate.reshape(-1, seqLen, inputs)
y_train=y_train[::seqLen].reshape(-1,1)
y_validate=y_validate[::seqLen].reshape(-1,1)

# train network
trained=False
#trained=True
if not trained:
    # for reproduce
    tf.set_random_seed(seed2)
    np.random.seed(0)

    biasInit= keras.initializers.Constant(value=1.)
    init= keras.initializers.TruncatedNormal(mean=0., stddev=0.05)
    model=Sequential()
    if hiddenLayers==1:
        model.add(LSTM(units, input_shape=(seqLen, inputs),
                       kernel_initializer=init,
                       bias_initializer=biasInit,
                       return_sequences=False))
        model.add(Dropout(dropoutRate))
    else:
        model.add(LSTM(units, input_shape=(seqLen, inputs),
                       kernel_initializer=init,
                       bias_initializer=biasInit,
                       return_sequences=True))
        model.add(Dropout(dropoutRate))
        for i in range(0,hiddenLayers-2):
            model.add(LSTM(units, 
                           kernel_initializer=init,
                           bias_initializer=biasInit,
                           return_sequences=True))
            model.add(Dropout(dropoutRate))
        model.add(LSTM(units, 
                       kernel_initializer=init,
                       bias_initializer=biasInit,
                       return_sequences=False))
        model.add(Dropout(dropoutRate))

    #model.add(Dense(units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optim=keras.optimizers.Adam(lr=learningRate)
    model.compile(loss='binary_crossentropy', 
                  optimizer=optim,
                  metrics=['accuracy'])

    model.summary()

    history=model.fit(x_train, y_train, validation_data=(x_validate, y_validate),
              #epochs=105, batch_size=16, shuffle=False)
              epochs=150, batch_size=16, shuffle=False)

    print('Finished Training')
    #model.save('model.h5')
    trained=True
    
    figureIndex=0
    plt.figure(figureIndex)
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    figureIndex+=1
    plt.figure(figureIndex)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

model=keras.models.load_model('model.h5')
print('load model successfully')
model.summary()
