#There are a lot of utils. Split function, Different RNN types, activations, optimizators etc. And also grid_generator, which provides iteration over search space (with embedded dicts)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
np.random.seed(5531)

from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, matthews_corrcoef
import sys
import json
import math
import time
from keras.models import model_from_json
from theano import tensor as T
import keras.backend as K
import copy
import gc
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta, Adagrad, Adam, RMSprop, Nadam, SGD
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
import h5py
from keras import regularizers

import keras.callbacks as ckbs
from keras.models import Model
from keras.layers import Input
K.set_image_dim_ordering('th')
from keras.layers import Concatenate
from sklearn.grid_search import ParameterGrid
import pickle
from keras.layers.recurrent import GRU, LSTM

from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces

def specificity(cm):
    return 1.0*cm[1,1]/(cm[1,0]+cm[1,1])

def sensitivity(cm):
    return 1.0*cm[0,0]/(cm[0,0]+cm[0,1])

def calc_metrics(true, pred):
    metrics = []
    metrics.append(f1_score(true, pred.round(), average='weighted'))
    metrics.append(roc_auc_score(true, pred))
    metrics.append(matthews_corrcoef(true, pred.round()))
    cm = confusion_matrix(true, pred.round())
    metrics.append(specificity(cm))
    metrics.append(sensitivity(cm))
    metrics.append(accuracy_score(true, pred.round()))
    return np.array(metrics)

def split_on_batches(X, y, bs = 40):
    lens = np.unique([len(x) for x in X])
    ldict = {l : [[x,y] for x,y in zip(X, y) if (len(x) == l)] for l in lens}
    bX = []
    by = []
    X_batch = []
    y_batch = []
    for l in lens:
        d = ldict[l]
        if (len(d) <= bs):
            bX.append([x[0] for x in d])
            by.append([x[1] for x in d])
        else:
            for i in range(0, len(d), bs):
                bX.append([x[0] for x in d[i:i+bs]])
                by.append([x[1] for x in d[i:i+bs]])
    return bX, by

def split_data(X, y, y4, yn):
    bX, by = split_on_batches(X, y, 32)
    _, by4 = split_on_batches(X, y4, 32)
    _, byn = split_on_batches(X, yn, 32)
    X_test,X_train,y_test,y_train = [],[],[],[]
    y_train_noise, y_test_noise = [],[]
    y_train_4, y_test_4 = [],[]
    for i,(x,y,yn,y4) in enumerate(zip(bX,by,byn,by4)):
        x = np.array(x)
        yn = np.array(yn)
        y = np.array(y)
        y4 = np.array(y4)
        X_train.append(x)
        y_train_noise.append(yn.reshape((yn.shape[0], yn.shape[1], 1)))
        y_train.append(y)
        y_train_4.append(y4)
    return X_train, y_train, y_train_4, y_train_noise

def get_rnn(name, units, act='tanh', retseq=True, inp=None):
    if (name == 'gru'):
        if (inp is not None):
            return GRU(units, return_sequences = retseq, activation=get_activation(act, True), input_shape=inp)
        else:
            return GRU(units, return_sequences = retseq, activation=get_activation(act, True))
    else:
        if (inp is not None):
            return LSTM(units, return_sequences = retseq, activation=get_activation(act, True), input_shape=inp)
        else:
            return LSTM(units, return_sequences = retseq, activation=get_activation(act, True))
        
    
def get_rnn_cell(name, units, inp=None, act='tanh'):
    if (name == 'gru'):
        if (inp is None):
            return GRUCell(units, activation=get_activation(act))
        else:
            return GRUCell(units, input_dim=inp, activation=get_activation(act))
    else:
        if (inp is None):
            return LSTMCell(units, activation=get_activation(act))
        else:
            return LSTMCell(units, input_dim=inp, activation=get_activation(act))

def get_optimizer(name, lr):
    if (name == 'adam'):
        return Adam(lr = lr)
    elif (name == 'adadelta'):
        return Adadelta(lr = lr)
    elif (name == 'adagrad'):
        return Adagrad(lr = lr)
    elif (name == 'rmsprop'):
        return RMSprop(lr = lr)
    elif (name == 'nadam'):
        return Nadam(lr = lr)
    elif (name == 'sgd'):
        return SGD(lr = lr)

def get_activation(act, string=False):
    str_act = ['relu', 'tanh', 'sigmoid', 'linear','softmax','softplus','softsign','hard_sigmoid']
    if (act in str_act):
        if string:
            return act
        else:
            return Activation(act)
    else:
        return {'prelu': PReLU(), 'elu' : ELU(), 'lrelu' : LeakyReLU(),
               }[act]
    
def get_reg(reg, l):
    if (reg == 'l1'):
        return regularizers.l1(l)
    elif (reg == 'l2'):
        return regularizers.l2(l)
    elif (reg == 'l1_l2'):
        return regularizers.l1_l2(l, l)
        
def grid_generator(search_space):
    param_grid = ParameterGrid(search_space)
    all_params = []
    for p in param_grid:
        all_params.append(p)
    for key in search_space.keys():
        if (isinstance(search_space[key], dict)):
            new_params=[]
            for param in all_params:
                if (search_space[key][param[key]] is None):
                    new_params.append(param)
                else:
                    param_grid = ParameterGrid(search_space[key][param[key]])
                    add_params = [p for p in param_grid]
                    for aparam in add_params:
                        tparam = copy.copy(param)
                        tparam.update(aparam)
                        new_params.append(tparam)
            all_params = new_params
    for param in all_params:
        yield param

def save_model(nn, path, i):
    pickle.dump([nn.to_json(), nn.get_weights()], open(path + "{}_model.pkl".format(i), 'wb'))
