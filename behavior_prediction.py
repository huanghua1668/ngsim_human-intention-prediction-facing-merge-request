import tensorflow as tf
#from tensorflow import keras
import  keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

primeNumber=1002527

def visualize_sample_sliced_by_velocity(samples, label):
    data=samples[:,1:]
    merge_after=samples[:,0].astype(int)
    #data0=data[label==0]
    data00=data[np.logical_and(label==0, merge_after==0)]
    data01=data[np.logical_and(label==0, merge_after==1)]
    data1=data[label==1]
    #ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='adv')
    for i in range(5):
        index00=np.logical_and(data00[:,0]>i*5., data00[:,0]<=(i+1)*5.)
        index01=np.logical_and(data01[:,0]>i*5., data01[:,0]<=(i+1)*5.)
        index1=np.logical_and( data1[:,0]>i*5.,  data1[:,0]<=(i+1)*5.)
        fig, ax=plt.subplots()
        ax.scatter(data00[index00,1],data00[index00,4],c='black', marker='x', 
                   label='adv(merge infront)')
        ax.scatter(data01[index01,1],data01[index01,4],c='red',   marker='x',
                   label='adv(merge after)')
        ax.scatter(data1[index1,1], data1[index1,4], c='blue',  marker='o', label='coop')

        plt.ylabel('$\Delta x$')
        plt.xlabel('$\Delta v$')
        #plt.axis([-15, 15, -50, 100])
        plt.axis([-7, 5, -10, 50])
        ax.legend()
        plt.title('$v_{ego}\leq$'+str(i*5+5))

    plt.show()

def visualize_sample(samples, label):
    fig, ax=plt.subplots()
    data=samples[:,1:]
    merge_after=samples[:,0].astype(int)
    #data0=data[label==0]
    data00=data[np.logical_and(label==0, merge_after==0)]
    data01=data[np.logical_and(label==0, merge_after==1)]
    data1=data[label==1]
    #ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='adv')
    ax.scatter(data00[:,1],data00[:,4],c='black', marker='x', label='adv(merge infront)')
    ax.scatter(data01[:,1],data01[:,4],c='red',   marker='x', label='adv(merge after)')
    ax.scatter(data1[:,1], data1[:,4], c='blue',  marker='o', label='coop')

    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    #plt.axis([-15, 15, -50, 100])
    plt.axis([-7, 2, 0, 50])
    ax.legend()
    plt.show()

def visualize_prediction(x_test0, label, predict):
    data=x_test0[:,1:]
    fig, ax=plt.subplots()

    data1=data[np.logical_and(label==1, predict==1)]
    data0=data[np.logical_and(label==1, predict==0)]
    ax.scatter(data1[:,1],data1[:,4],c='blue', marker='o', label='true coop')
    ax.scatter(data0[:,1],data0[:,4],c='blue', marker='x', label='false adv')

    merge_after=x_test0[:,0].astype(int)
    data00=data[np.logical_and(np.logical_and(label==0, predict==0),
                                             merge_after==0)]
    data01=data[np.logical_and(np.logical_and(label==0, predict==0),
                                             merge_after==1)]
    data10=data[np.logical_and(np.logical_and(label==0, predict==1),
                                             merge_after==0)]
    data11=data[np.logical_and(np.logical_and(label==0, predict==1),
                                             merge_after==1)]
    ax.scatter(data00[:,1],data00[:,4],c='black', marker='o', label='true adv(merge infront)')
    ax.scatter(data01[:,1],data01[:,4],c='red',   marker='o', label='true adv(merge after)')
    ax.scatter(data10[:,1],data10[:,4],c='black', marker='x', label='false coop(merge infront)')
    ax.scatter(data11[:,1],data11[:,4],c='red',   marker='x', label='false coop(merge after)')


    #for i in range(data.shape[0]):
    #    y=label[i]
    #    if y==1:
    #    if y==1:
    #        if predict[i]==1:
    #            plt.scatter(data[i][1],data[i][4],color='blue', marker='o')
    #        else:
    #            plt.scatter(data[i][1],data[i][4],color='blue', marker='x')
    #    else:
    #        if predict[i]==1:
    #            if data0[i,0]==1: # merge after
    #                plt.scatter(data[i][1],data[i][4],color='red', marker='x')
    #            else:
    #                plt.scatter(data[i][1],data[i][4],color='black', marker='x')
    #        else:
    #            if data0[i,0]==1: # merge after
    #                plt.scatter(data[i][1],data[i][4],color='red', marker='o')
    #            else:
    #                plt.scatter(data[i][1],data[i][4],color='black', marker='o')

    plt.ylabel('$\Delta x$')
    plt.xlabel('$\Delta v$')
    #plt.axis([-15, 15, -50, 100])
    plt.axis([-7, 2, 0, 50])
    ax.legend()
    plt.show()

def preprocess():
    # the data, split between train and test sets
    datas=[]
    datas.append(np.genfromtxt('0400pm-0415pm/samples_3seconds_begin_after.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_3seconds_begin_after.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_3seconds_begin_after.csv', delimiter=','))
    datas.append(np.genfromtxt('0400pm-0415pm/samples_merge_after.csv', delimiter=','))
    datas.append(np.genfromtxt('0500pm-0515pm/samples_merge_after.csv', delimiter=','))
    datas.append(np.genfromtxt('0515pm-0530pm/samples_merge_after.csv', delimiter=','))

    for i in range(3):
        #0 for merge in front
        temp=np.zeros((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp
    for i in range(3,6):
        #1 for merge after
        temp=np.ones((datas[i].shape[0], datas[i].shape[1]+1))
        temp[:,1:]=datas[i]
        datas[i]=temp

    data=np.vstack((datas[0], datas[1]))
    for i in range(2,6):
        data=np.vstack((data, datas[i]))
    ys=data[:,-1]
    print(ys.shape[0], ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')

    np.random.seed(primeNumber)
    np.random.shuffle(data)

    #for i in range(10):
    #    print(data[i])

    data_train=[]
    for i in range(0, data.shape[0]-4, 5):
        data_train.append(data[i])
        data_train.append(data[i+1])
        data_train.append(data[i+2])
    sc_X = StandardScaler()
    data_train=np.vstack(data_train)
    np.savez("train.npz",a=data_train)
    x_train=data_train[:, 1:-1]
    y_train=data_train[:, -1].astype(int)
    #visualize_sample(data_train[:, 0:-1] , y_train)
    #visualize_sample_sliced_by_velocity(data_train[:,0:-1], y_train)
    x_train=sc_X.fit_transform(x_train)

    data_validate=data[3::5]
    np.savez("validate.npz",a=data_validate)
    x_validate=data_validate[:,1:-1]
    x_validate=sc_X.transform(x_validate)
    y_validate=data_validate[:,-1].astype(int)
    data_test=data[4::5]
    np.savez("test.npz",a=data_test)
    x_test=data_test[:,1:-1]
    x_test=sc_X.transform(x_test)
    y_test=data_test[:,-1].astype(int)
    print(y_train.shape[0], ' trainning samples, and ', 
          np.sum(y_train)/y_train.shape[0]*100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ', 
          np.sum(y_validate)/y_validate.shape[0]*100, '% positives')
    print(y_test.shape[0], ' test samples, and ', 
          np.sum(y_test)/y_test.shape[0]*100, '% positives')
    return x_train, x_validate, x_test, y_train, y_validate, y_test, sc_X

def preprocess_multiple():
    # the data, split between train and test sets
    data0=np.genfromtxt('0400pm-0415pm/samples_multiple_snapshot.csv', delimiter=',')
    data1=np.genfromtxt('0500pm-0515pm/samples_multiple_snapshot.csv', delimiter=',')
    data2=np.genfromtxt('0515pm-0530pm/samples_multiple_snapshot.csv', delimiter=',')
    data=np.vstack((data0, data1))
    data=np.vstack((data, data2))
    ys=data[:,-1]
    print(ys.shape[0], ' samples, and ', np.sum(ys)/ys.shape[0]*100, '% positives')
    
    data_train=[]
    data_validate=[]
    data_test=[]
    mixture=False
    for i in range(data.shape[0]):
        if not mixture:
            laneChange=int(data[i,0])
            if laneChange%5<3:
                data_train.append(data[i,1:])
            elif laneChange%5==3:
                data_validate.append(data[i,1:])
            else:
                data_test.append(data[i,1:])
        else:
            if i%5<3:
                data_train.append(data[i,1:])
            elif i%5==3:
                data_validate.append(data[i,1:])
            else:
                data_test.append(data[i,1:])
    sc_X = StandardScaler()
    data_train=np.vstack(data_train)
    x_train=data_train[:, :-1]
    x_train=sc_X.fit_transform(x_train)
    y_train=data_train[:, -1].astype(int)

    data_validate=np.vstack(data_validate)
    x_validate=data_validate[:,:-1]
    x_validate=sc_X.transform(x_validate)
    y_validate=data_validate[:,-1].astype(int)

    data_test=np.vstack(data_test)
    x_test=data_test[:,:-1]
    x_test=sc_X.transform(x_test)
    y_test=data_test[:,-1].astype(int)
    print(y_train.shape[0], ' trainning samples, and ', 
          np.sum(y_train)/y_train.shape[0]*100, '% positives')
    print(y_validate.shape[0], ' validate samples, and ', 
          np.sum(y_validate)/y_validate.shape[0]*100, '% positives')
    print(y_test.shape[0], ' test samples, and ', 
          np.sum(y_test)/y_test.shape[0]*100, '% positives')
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def analysis(x_test0, y_predict, y_test):
    visualize_prediction(x_test0, y_test, y_predict)

    #for merge after
    label=y_test[x_test0[:,0]==1]
    predict=y_predict[x_test0[:,0]==1]
    accuracy=np.sum(label==predict)/label.shape[0]
    print('accuracy for merge after samples', accuracy)
    #for merge infront
    label=y_test[x_test0[:,0]==0]
    predict=y_predict[x_test0[:,0]==0]
    accuracy=np.sum(label==predict)/label.shape[0]
    print('accuracy for merge infront samples', accuracy)

    accuracy=np.sum(y_predict==y_test)/y_predict.shape[0]
    truePos=np.sum(y_test)
    trueNeg=y_test.shape[0]-np.sum(y_test)
    predictedPos=np.sum(y_predict)
    predictedNeg=y_predict.shape[0]-np.sum(y_predict)
    print('true positive ', truePos)
    print('true negative ', trueNeg)
    print('predicted positive ', predictedPos)
    print('predicted negative ', predictedNeg)

def feature_extract(x_train0, x_validate0, x_test0):
    x_selected=[]
    #x_train=      x_train0[:,[0,1,4]]
    #x_validate=x_validate0[:,[0,1,4]]
    #x_test=        x_test0[:,[0,1,4]]
    x_train=      x_train0[:,:7]
    x_validate=x_validate0[:,:7]
    x_test=        x_test0[:,:7]
    x_selected.append(x_train)
    x_selected.append(x_validate)
    x_selected.append(x_test)
    return x_selected

def train():
    inputs=10
    batch_size = 64
    epochs = 200
    seed0=1013899
    seed1=1018859
    units=64
    learningRate=0.0002
    hiddenLayers=3
    
    #x_train, x_validate, x_test, y_train, y_validate, y_test=preprocess_multiple()
    x_train, x_validate, x_test, y_train, y_validate, y_test, sc_X=preprocess()
    # features=[u, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2]
    print(x_train.shape[0], 'train samples')
    print(x_validate.shape[0], 'validate samples')
    print(x_test.shape[0], 'test samples')

    # eliminate features of dy1, dy2
    #x_train=x_train[:,:-3]
    #x_validate=x_validate[:,:-3]
    #x_test=x_test[:,:-3]
    
    # convert class vectors to binary class matrices
    
    init= keras.initializers.TruncatedNormal(mean=0., stddev=0.05, seed=seed0)
    regularizer=keras.regularizers.l2(0.00)
    
    model = Sequential()
    model.add(Dense(units, kernel_initializer=init, kernel_regularizer=regularizer,
                    activation='relu', input_shape=(inputs,)))
    #hh: here use_bias=False, so no bias
    #model.add(Dropout(0.2))
    #hh: Dropout consists in randomly setting a fraction rate of input
    #    units to 0 at each update during training time, which helps prevent
    #    overfitting.
    for i in range(hiddenLayers):
        model.add(Dense(units,  kernel_initializer=init, kernel_regularizer=regularizer,
                        activation='relu'))
        #model.add(Dropout(0.05, seed=seed1))
    model.add(Dense(1,  kernel_initializer=init, kernel_regularizer=regularizer,
                    activation='sigmoid'))
    
    model.summary()
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learningRate),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,                                         
                        validation_data=(x_validate, y_validate))
                        #validation_data=(x_test, y_test))
    model.save('model.h5')
    #model=keras.models.load_model('model.h5')
    #print('load model successfully')
    #model.summary()
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
   
    score=model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
          
    # visualize prediction
    f=np.load('test.npz')
    x_test0=f['a']
    y_predict=model.predict_classes(x_test)
    y_predict=y_predict[:,0]
    print('x_test0 shape', x_test0.shape)
    analysis(x_test0, y_predict, y_test)

    f=np.load('train.npz')
    x_train0=f['a']
    y_predict=model.predict_classes(x_train)
    y_predict=y_predict[:,0]
    print('x_train0 shape', x_train0.shape)
    analysis(x_train0, y_predict, y_train)

def preprocess_sequence(sc_X, i):
    # sc_X is the transform method
    # i is event ID
    data0=np.genfromtxt('0400pm-0415pm/samples_snapshots.csv', delimiter=',')
    #data0=np.genfromtxt('0400pm-0415pm/samples_merge_after_snapshots.csv', delimiter=',')
    data0=data0[data0[:,0]==i][:,1:]
    dt=data0[:,0]
    data=data0[:,1:-1]
    xs=sc_X.transform(data)
    label=data0[0,-1]
    return xs, data0[:,1:-1], label,dt

def predict_sequences():
    inputs=10
    batch_size = 64
    epochs = 200
    seed0=1013899
    seed1=1018859
    units=64
    learningRate=0.0002
    hiddenLayers=3
    
    #x_train, x_validate, x_test, y_train, y_validate, y_test=preprocess_multiple()
    x_train, x_validate, x_test, y_train, y_validate, y_test, sc_X=preprocess()
    # features=[u, du0, du1, du2, dx0, dx1, dx2, dy0, dy1, dy2]
    print(x_train.shape[0], 'train samples')

    # eliminate features of dy1, dy2
    #x_train=x_train[:,:-3]
    #x_validate=x_validate[:,:-3]
    #x_test=x_test[:,:-3]
    
    #model.save('model.h5')
    model=keras.models.load_model('model.h5')
    print('load model successfully')
    model.summary()
   
    for case in range(200):
    #for case in cases:
        #case=10
        xs,xs0,label,dt=preprocess_sequence(sc_X, case)
        y_predict=model.predict(xs)
        #if (y_predict[0]<0.5 and label==0) or (y_predict[0]>=0.5 and label==1):
        if (y_predict.shape[0]>9 and 
            ((y_predict[-10]<0.5 and label==0) or (y_predict[-10]>=0.5 and label==1))):
            continue
        print(case)
        print('label:', label)
   
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.plot(dt, xs0[:,1], xs0[:,4])
        #ax.set_xlabel('t(s)')
        #ax.set_ylabel('$dv$')
        #ax.set_zlabel('$dx$')
        #
        ## visualize prediction
        ##y_predict=model.predict_classes(xs)
        #y_predict=y_predict[:,0]

        #plt.figure()
        #plt.plot(dt, y_predict)
        #plt.xlabel('t(s)')
        #plt.ylabel('p(coop)')
        #plt.show()

def train_selected_features():
    # try to see if only [ve, dx0, dv0] will improve test performance
    #inputs=3
    inputs=7
    batch_size = 16
    epochs = 100
    seed0=1013899
    seed1=1018859
    units=64
    learningRate=0.0002
    hiddenLayers=4
    
    #x_train, x_validate, x_test, y_train, y_validate, y_test=preprocess_multiple()
    x_train, x_validate, x_test, y_train, y_validate, y_test, sc_X=preprocess()
    print(x_train.shape[0], 'train samples')
    print(x_validate.shape[0], 'validate samples')
    print(x_test.shape[0], 'test samples')

    # eliminate features of dy1, dy2
    #x_train=x_train[:,:-3]
    #x_validate=x_validate[:,:-3]
    #x_test=x_test[:,:-3]
    
    # convert class vectors to binary class matrices
    
    init= keras.initializers.TruncatedNormal(mean=0., stddev=0.05, seed=seed0)
    regularizer=keras.regularizers.l2(0.00)
    
    model = Sequential()
    model.add(Dense(units, kernel_initializer=init, kernel_regularizer=regularizer,
                    activation='relu', input_shape=(inputs,)))
    #hh: here use_bias=False, so no bias
    #model.add(Dropout(0.2))
    #hh: Dropout consists in randomly setting a fraction rate of input
    #    units to 0 at each update during training time, which helps prevent
    #    overfitting.
    for i in range(hiddenLayers):
        model.add(Dense(units,  kernel_initializer=init, kernel_regularizer=regularizer,
                        activation='relu'))
        #model.add(Dropout(0.05, seed=seed1))
    model.add(Dense(1,  kernel_initializer=init, kernel_regularizer=regularizer,
                    activation='sigmoid'))
    
    model.summary()
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=learningRate),
                  metrics=['accuracy'])

    # select features
    x_selected=feature_extract(x_train, x_validate, x_test)
    x_train=x_selected[0]
    x_validate=x_selected[1]
    x_test=x_selected[2]

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,                                         
                        validation_data=(x_validate, y_validate))
                        #validation_data=(x_test, y_test))
    #model.save('model.h5')
    #model=keras.models.load_model('model.h5')
    #print('load model successfully')
    #model.summary()
    
    figureIndex=0
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    
    plt.figure(figureIndex)
    figureIndex+=1
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
   
    score=model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
          
    # visualize prediction
    f=np.load('test.npz')
    x_test0=f['a']
    y_predict=model.predict_classes(x_test)
    y_predict=y_predict[:,0]
    print('x_test0 shape', x_test0.shape)
    analysis(x_test0, y_predict, y_test)

    f=np.load('train.npz')
    x_train0=f['a']
    y_predict=model.predict_classes(x_train)
    y_predict=y_predict[:,0]
    print('x_train0 shape', x_train0.shape)
    analysis(x_train0, y_predict, y_train)

train()
#predict_sequences()
