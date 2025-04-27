
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add 
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input
from sklearn.metrics import confusion_matrix,roc_curve,precision_recall_curve,auc
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,f1_score,precision_score, classification_report
from sklearn.model_selection import StratifiedKFold ,StratifiedShuffleSplit
import random
import os
from decimal import Decimal
import numpy as np
import pandas as pd

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
    dataset = pd.read_csv('validationDataset.csv')
    X_test = dataset[dataset.columns[2:len(dataset.columns)-1]].values
    y_test = dataset['98'].values
    y_test_noresh = transy(y_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    test_prob = model.predict(X_test)
    test_predict=(np.asarray(test_prob)).round()
    test_predict = test_predict.reshape((test_predict.shape[0],  test_predict.shape[2]))
    test_prob = test_prob.reshape((test_prob.shape[0],  test_prob.shape[2]))    
    target_names = ['normal', 'diabetes']
    crdict = classification_report(y_test_noresh, test_predict,target_names=target_names)
    print(crdict)
