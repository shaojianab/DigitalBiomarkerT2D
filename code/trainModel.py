
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers.core import Lambda
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Model
from keras import optimizers
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,f1_score,precision_score, classification_report
from sklearn.model_selection import StratifiedKFold ,StratifiedShuffleSplit
import random
import os
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from decimal import Decimal

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def _bn_relu(layer, dropout=0, **params):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D 
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('softmax')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

def build_network(**params):
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'])

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)
    return model

def build_model():
    
    model = build_network(input_shape = [96,1],
                          num_categories = 2,
        conv_subsample_lengths= [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,1, 2, 1],
    conv_filter_length= 11,
    conv_num_filters_start= 32,
    conv_init= 'he_normal',
    conv_activation= 'relu',
    conv_dropout= 0.2,
    conv_num_skip= 2,
    conv_increase_channels_at= 4,
    is_regular_conv =  False,
    learning_rate= 0.001,
    batch_size= 128)
    print(model.summary())   
    return model


def transy(y):
    y_label = []
    for i in y:
        if i ==0:
            y_label.append([1,0])
        else:
            y_label.append([0,1])
    y_label = np.asarray(y_label)
    return y_label
    

if __name__ == '__main__':
    traindataset = pd.read_csv('train.csv')
    devdataset = pd.read_csv('dev.csv')
    testdataset = pd.read_csv('test.csv')
    
    columns = traindataset.columns[12:len(traindataset.columns)-1].tolist()
    for c in columns:
        traindataset[c] = traindataset[c]/18
       
    columns = devdataset.columns[12:len(devdataset.columns)-1].tolist()
    for c in columns:
        devdataset[c] = devdataset[c]/18
        
    columns = testdataset.columns[12:len(testdataset.columns)-1].tolist()
    for c in columns:
        testdataset[c] = testdataset[c]/18
        
    X_train = traindataset[traindataset.columns[12:len(traindataset.columns)-1]].values
    y_train = traindataset['diabtype'].values
    
    X_dev = devdataset[devdataset.columns[12:len(devdataset.columns)-1]].values
    y_dev = devdataset['diabtype'].values
    
    X_test = testdataset[testdataset.columns[12:len(testdataset.columns)-1]].values
    y_test = testdataset['diabtype'].values
    
    y_train_noresh = transy(y_train)
    y_train_resh = transy(y_train)
    y_dev_noresh = transy(y_dev)
    y_dev_resh = transy(y_dev)
    y_test_noresh = transy(y_test)
    y_test_resh = transy(y_test)


    y_train_resh = y_train_resh.reshape((y_train_resh.shape[0], 1,  y_train_resh.shape[1]))
    y_dev_resh = y_dev_resh.reshape((y_dev_resh.shape[0], 1,  y_dev_resh.shape[1]))
    y_test_resh = y_test_resh.reshape((y_test_resh.shape[0],  1, y_test_resh.shape[1]))    
    X_train = X_train.reshape((X_train.shape[0],  X_train.shape[1], 1))
    X_dev = X_dev.reshape((X_dev.shape[0],  X_dev.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    
    history = LossHistory()
    
    model = build_model()
    reduce_lr = ReduceLROnPlateau(
            factor=0.1,
            patience=2,
            min_lr= 0.001 * 0.001)
    stopping = EarlyStopping(patience=10, monitor = 'val_loss')
    model.fit(x=X_train, y=y_train_resh, validation_data=(X_dev, y_dev_resh),epochs=500, batch_size=256, callbacks = [reduce_lr, stopping,history])  
    modelfpath = '/home/ryan/cgm_classification_diabetes/lstmmodel/' + 'ada_scnn_userlevel' + '.hdf5'
    model.save(modelfpath, overwrite=True)
    
    
    train_predict_prob= model.predict(X_train)
    train_predict=(np.asarray(train_predict_prob)).round()
    train_predict = train_predict.reshape((train_predict.shape[0],  train_predict.shape[2]))
    train_predict_prob = train_predict_prob.reshape((train_predict_prob.shape[0],  train_predict_prob.shape[2]))
    #test_predict=model.predict(x_test_pad)
    target_names = ['normal', 'diabetes']
    crdict = classification_report(y_train_noresh, train_predict,target_names=target_names)
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train_noresh[:,1], train_predict_prob[:,1], pos_label=1)
    print(metrics.auc(fpr_train, tpr_train))
    print(crdict)    
    
    dev_predict_prob= model.predict(X_dev)
    dev_predict=(np.asarray(dev_predict_prob)).round()
    dev_predict = dev_predict.reshape((dev_predict.shape[0],  dev_predict.shape[2]))
    dev_predict_prob = dev_predict_prob.reshape((dev_predict_prob.shape[0],  dev_predict_prob.shape[2]))
    target_names = ['normal', 'diabetes']
    crdict = classification_report(y_dev_noresh, dev_predict,target_names=target_names)
    fpr_dev, tpr_dev, thresholds_dev = metrics.roc_curve(y_dev_noresh[:,1], dev_predict_prob[:,1], pos_label=1)
    print(metrics.auc(fpr_dev, tpr_dev))
    print(crdict) 
    
    test_predict_prob= model.predict(X_test)
    test_predict=(np.asarray(test_predict_prob)).round()
    test_predict = test_predict.reshape((test_predict.shape[0],  test_predict.shape[2]))
    test_predict_prob = test_predict_prob.reshape((test_predict_prob.shape[0],  test_predict_prob.shape[2]))

    target_names = ['normal', 'diabetes']
    crdict = classification_report(y_test_noresh, test_predict,target_names=target_names)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_noresh[:,1], test_predict_prob[:,1], pos_label=1)
    print(metrics.auc(fpr, tpr))
    print(crdict)
    
    my_dataset = pd.read_csv('/home/ryan/cgm_classification_diabetes/dataset/dataset_diabetes_classification_day.csv')

    my_train = my_dataset[my_dataset.columns[2:len(my_dataset.columns)-1]].values
    my_y_train = my_dataset['98'].values

    my_y_train = transy(my_y_train)

    my_train = my_train.reshape((my_train.shape[0],  my_train.shape[1], 1))

    my_test_prob = model.predict(my_train)
    my_test_predict=(np.asarray(my_test_prob)).round()
    my_test_predict = my_test_predict.reshape((my_test_predict.shape[0],  my_test_predict.shape[2]))
    
    my_test_prob = my_test_prob.reshape((my_test_prob.shape[0],  my_test_prob.shape[2]))
    
    target_names = ['normal', 'diabetes']
    crdict = classification_report(my_y_train, my_test_predict,target_names=target_names)
    my_fpr, my_tpr, my_thresholds = metrics.roc_curve(my_y_train[:,1], my_test_prob[:,1], pos_label=1)
    print(metrics.auc(my_fpr, my_tpr))    
    print(crdict)
  
    history.loss_plot('epoch')
