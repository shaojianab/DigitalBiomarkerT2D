from keras import backend as K
from keras.layers import BatchNormalization
from imblearn.over_sampling import RandomOverSampler
from keras.layers import Activation
from keras.layers import Add 
from keras.layers import MaxPooling1D
from keras.layers.core import Lambda
from keras.layers.core import Dense, Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Model,load_model
import matplotlib.pyplot as plt
from inspect import signature
from keras.layers import Input
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,auc
from keras.optimizers import Adam
import math
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.layers.core import *
from keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,f1_score,precision_score, classification_report
from sklearn.model_selection import StratifiedKFold ,StratifiedShuffleSplit
import random
import os
from decimal import Decimal


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

    model = load_model('model.hdf5')
    testdataset = pd.read_csv('test.csv')

    columns = testdataset.columns[12:len(testdataset.columns)-1].tolist()
    for c in columns:
        testdataset[c] = testdataset[c]/18
    
    X_test = testdataset[testdataset.columns[12:len(testdataset.columns)-1]].values
    y_test = testdataset['diabtype'].values


    y_test_noresh = transy(y_test)
 
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    test_prob = model.predict(X_test)
    test_predict=(np.asarray(test_prob)).round()
    
    test_predict = test_predict.reshape((test_predict.shape[0],  test_predict.shape[2]))
    test_prob = test_prob.reshape((test_prob.shape[0],  test_prob.shape[2]))    
    #print(y_test)
    target_names = ['normal', 'diabetes']
    crdict = classification_report(y_test_noresh, test_predict,target_names=target_names,output_dict=True)
    print(crdict)
    
    
    
