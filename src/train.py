import numpy as np
np.random.seed(5531)
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help='Gpu #NUM')
parser.add_argument("--exp_path", type=str, help='Path for experiment information save to')
parser.add_argument("--data_path", type=str, help='Data path')
parser.add_argument("--cv_path", type=str, help='CV split file path')
args = parser.parse_args()
import os                                                                                                                                                               
os.environ["THEANO_FLAGS"] = "floatX=float32, device=gpu{0}, lib.cnmem={1}".format(args.gpu, 0.35) 
import warnings
warnings.filterwarnings("ignore")
import sys
from keras.models import model_from_json
import copy
import gc
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.optimizers import Adadelta, Adagrad, Adam, RMSprop, Nadam
from copy import deepcopy
from keras.layers import Input, concatenate, RepeatVector
from keras.models import Model
from keras import backend as K
from utils import *
K.set_image_dim_ordering('th')
import pickle
from keras.layers.wrappers import TimeDistributed
from keras.utils.np_utils import to_categorical
from sklearn.metrics import matthews_corrcoef, accuracy_score

#get accuracy for binary predicts
def score_seq(model, xt, yt, ind):
    pred = []
    true = []
    for y in yt:
        true += list(y.flatten())
    for x, y in zip(xt, yt):
        pr = model.predict(x)[ind]
        pred += list(pr.flatten().round())
    try:
        mat = matthews_corrcoef(true, pred)
    except:
        mat = 0
    return mat  

#get accuracy for 4-class predicts
def score_seq4(model, xt, yt):
    pred = []
    true = []
    for y in yt:
        y = np.array([np.argmax(p) for p in y])
        true += list(y.flatten())
    for x, y in zip(xt, yt):
        pr = model.predict(x)[1]
        pr = np.array([np.argmax(p) for p in pr])
        pred += list(pr.flatten().round())
    mat = accuracy_score(true, pred)
    return mat   

class RNN():
    def __init__(self, params):
        self.params = params
        self.model = self.build(params)
        
    #build the model for 4-class classification
    def build(self, params):
        #Noise RNN
        inp = Input((None, 130,))
        m = get_rnn(params['rnn_type'], params['units'], params['activation'])(inp)
        for i in range(params['deep'] - 1):
            m = get_rnn(params['rnn_type'], params['units'], params['activation'])(m)
        noise = TimeDistributed(Dense(1, activation='sigmoid'))(m)
        #Noise Masking
        inp2 = Lambda(lambda x: (1-x[0])*x[1], output_shape=(None,130))([noise, inp])
        #Anomaly RNN
        m1 = get_rnn(params['rnn_type'], params['units'], params['activation'], params['deep'] > 1)(inp2)
        for i in range(params['deep'] - 1):
            m1 = get_rnn(params['rnn_type'], params['units'], params['activation'], params['deep'] > i+2)(m1)
        m1 = Dense(4, activation='softmax')(m1)
        model = Model(inp, [noise, m1])
        model.compile(loss=['binary_crossentropy','categorical_crossentropy'], loss_weights=[0.3, 0.7], optimizer=get_optimizer(params['optimizer'],params['lr']))
        return model
    
    #train method. provides training of the model batch by batch (because all batches has different lengths as sound samples)
    def train(self, X_train, X_test, y_train, y_test, y_train_noise, y_test_noise, epochs):
        self.losses = []
        best_test_m = 0
        best_test_m_noise = 0
        best_weights = None
        for epoch in range(epochs):
            for x,y,yn in zip(X_train, y_train, y_train_noise):
                self.model.train_on_batch(np.array(x), [np.array(yn), np.array(y)])
            test_m = score_seq4(self.model, X_test, y_test)
            test_m_noise = score_seq(self.model, X_test, y_test_noise, 0)
            if (test_m > best_test_m):
                best_test_m = test_m
                best_test_m_noise = test_m_noise
                best_weights = self.model.get_weights()
            train_m = score_seq(self.model, X_train, y_train, 1)
            train_m_noise = score_seq(self.model, X_train, y_train_noise, 0)
            self.losses.append([test_m, train_m, test_m_noise, train_m_noise])
        if (best_weights is not None):
            self.model.set_weights(best_weights)
        return self.losses, best_test_m, best_test_m_noise
    
def get_predicts(model, xt, yt, ind):
    pred = []
    true = []
    for y in yt:
        true += list(y.flatten())
    for x, y in zip(xt, yt):
        pr = model.predict(x)[ind]
        pred += list(pr.flatten().round())
    return np.array(pred), np.array(true)
    
def add_experiment(X_train, X_test, y_train, y_test, y_train_noise, y_test_noise, params):
    rnn = RNN(params)
    losses, bm, bmn = rnn.train(X_train, X_test, y_train, y_test, y_train_noise, y_test_noise, params['epochs'])
    gc.collect()
    return rnn, losses, bm, bmn

#this method provides gridsearch over search space using cross-validation
def gridsearch(X_cv, y_cv, y4_cv, yn_cv,S, experiment_path, search_space, cv):
    if (not os.path.exists(experiment_path)):
        os.mkdir(experiment_path)
    stats = pd.DataFrame(columns=['Model id', 'Optimizer','LR', 'RNN type', 'RNN act', 'RNN units', 'Deep', 'Accuracy Anom', 'Matthew Noise'])
    history_filename = experiment_path + 'hist.h'
    grid = grid_generator(search_space)
    stats_filename = experiment_path + 'stats.csv'
    histories = {}
    i = 1
    for params in grid:
        print(i)
        print('-'*15)
        cur_exp_path = experiment_path + "/{}_model/".format(i)
        if (not os.path.exists(cur_exp_path)):
            os.mkdir(cur_exp_path)
        for j,k in enumerate(cv):
            tr, te = k[0], k[1]
            tr = [S[t] for t in tr]
            te = [S[t] for t in te]
            #split data
            X_train, y_train, y4_train, yn_train = X_cv[tr], y_cv[tr], y4_cv[tr], yn_cv[tr]
            X_test, y_test, y4_test, yn_test = X_cv[te], y_cv[te], y4_cv[te], yn_cv[te]
            y4_test = to_categorical(y4_test)
            y4_train = to_categorical(y4_train)
            X_test, y_test, y4_test, yn_test = split_data(X_test, y_test, y4_test, yn_test)
            X_train, y_train, y4_train, yn_train = split_data(X_train, y_train, y4_train, yn_train)
            #train model
            nn, losses, bm, bmn = add_experiment(X_train, X_test, y4_train, y4_test, yn_train, yn_test, params)
            #save model
            save_model(nn.model, cur_exp_path, j)
            #save metrics and parameters
            stats.loc[stats.shape[0]] = [i, params['optimizer'], params['lr'], params['rnn_type'], params['activation'], params['units'], params['deep'], bm, bmn]
            #save learning history
            histories[str(i)+"_"+str(j)] = losses
            pickle.dump(histories, open(history_filename, 'wb'))
            stats.to_csv(stats_filename, index=False)
        i += 1
    
if __name__ == '__main__':
    #load data and cv split
    X, y, y4, yn, S  = pickle.load(open(args.data_path, 'rb')) 
    cv = pickle.load(open(args.cv_path, 'rb'))
    #setup hyperparameters grid
    search_space = {'optimizer' : {'adam' : {'lr' : [0.0001]}},
                 'activation' : ['tanh'],
                 'epochs' : [1],
                 'units' : [256],
                 'deep' : [2],
                 'rnn_type':['gru']
                  }
    gridsearch(X, y, y4, yn, S, args.exp_path, search_space, cv)
