import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, GRU
from dataRetriever import DataRetriever
from dataPreprocessor import preprocessData, customNormalizer
from augmentData import augmentData

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, max_error
import tqdm

import sys
import os
import shutil

model_config = sys.argv[1] if len(sys.argv) > 0 else 'blstm3feats'
numdays = int(sys.argv[2]) if len(sys.argv) > 1 else 90

if not os.path.exists('./checkpoints/'):
    os.mkdir('./checkpoints/')

seed = 0  # np.random.randint(0, 100000)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_pos = pd.read_csv('data/jj6z-iyrp/2020-12-30.csv', index_col=0)
data_death = pd.read_csv('data/uqk7-bf9s/2020-12-30.csv', index_col=0)
dataset = preprocessData(data_pos, data_death)
normalizer = customNormalizer()
data, dates = normalizer.normalizeData(dataset)

look_back = 8
lstmdata = np.vstack((np.zeros((look_back, data.shape[1])), data))
X = []
Y = np.vstack((np.zeros((look_back, 3)), data[:,[0,1,-1]]))
for i in range(look_back):
    X.append(np.roll(lstmdata, i, axis=0))
X = np.array(X).transpose((1,0,2))[look_back:-1]
Y = np.array(Y)[look_back+1:]

X = X[:,:,[0, 1, -1]]  # only the 3 predicted labels

def create_model():
    inp = Input((look_back, X.shape[2]))  # X.shape[2] is 92 by default
    if 'blstm' in model_config:
        lstm = Bidirectional(LSTM(64), input_shape=(look_back, X.shape[2]))(inp)
    elif 'lstm' in model_config:
        lstm = LSTM(64, input_shape=(look_back, X.shape[2]))(inp)
    else:
        raise ValueError('network type not supported')

    if 'hidden' in model_config:
        lstm = Dense(100, activation='relu')(lstm)

    pos = Dense(1, name='positive')(lstm)
    dea = Dense(1, name='deaths')(lstm)
    r0 = Dense(1, name='r0')(lstm)
    model = Model(inputs=inp, outputs=[pos, dea, r0])
    return model

model = create_model()
model.load_weights(f'checkpoints/blstm3feats.best.h5')

y_pred = model.predict(X)
y_pred = np.array(y_pred).squeeze(-1).transpose()

prev_sample = np.vstack([y_pred[-1][None, :], X[-1, :-1]])

for day in range(numdays):
    future_day = model.predict(prev_sample[None,:])
    future_day = np.array(future_day).squeeze(-1).transpose()
    prev_sample = np.vstack([future_day, prev_sample[:-1]])
    y_pred = np.vstack([y_pred, future_day])

import datetime
base = datetime.datetime.strptime('2020-'+dates[0], "%Y-%m-%d")
date_list = [base + datetime.timedelta(days=x) for x in range(y_pred.shape[0])]
date_list = list(map(lambda x: x.strftime("%Y-%m-%d"), date_list))

y_labels = ['infected', 'deaths', 'R0']
f, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
for i in range(Y.shape[1]):
    ax[i].plot(y_pred[:,i], c='g')
    ax[i].plot(Y[:, i], c='k')
    ax[i].set_ylabel(y_labels[i])
    ax[i].grid('on')
plt.xticks(range(0, len(date_list[1:]), 7), date_list[1::7], rotation=90)
plt.xlabel('date')
plt.savefig(f'../images/{model_config}.{numdays}extension.png')
plt.show()