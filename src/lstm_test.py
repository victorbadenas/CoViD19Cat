import tensorflow as tf
import keras
import numpy as np
import pandas as pd
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
model_config = sys.argv[1]

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

def augmentData(X:np.ndarray, Y:np.ndarray):
    assert X.shape[0] == Y.shape[0], "arrays are inconsistant. axis 0 must be of the same size"

    Xextended = X.copy()
    Yextended = Y.copy()
    for std in [1e-4, 5e-5]:
        posRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,look_back))
        deaRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,look_back))
        r0Random = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,look_back))
        otherRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0],look_back,X.shape[2]-3))

        Xrand = np.concatenate([posRandom[:-1, :, None], deaRandom[:-1, :, None], otherRandom, r0Random[:-1, :, None]], axis=2)
        Yrand = np.concatenate([posRandom[1:, 0, None], deaRandom[1:, 0, None], r0Random[1:, 0, None]], axis=1)
        Xextended = np.concatenate((Xextended, X + Xrand))
        Yextended = np.concatenate((Yextended, Y + Yrand))
    return Xextended, Yextended

import tensorflow.keras.backend as K
def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

X, Y = augmentData(X, Y)

def create_model():
    inp = Input((look_back, 92))
    if 'blstm' in model_config:
        lstm = Bidirectional(LSTM(64), input_shape=(look_back, 92))(inp)
    elif 'lstm' in model_config:
        lstm = LSTM(64, input_shape=(look_back, 92))(inp)
    else:
        raise ValueError('network type not supported')

    if 'hidden' in model_config:
        lstm = Dense(100, activation='relu')(lstm)

    pos = Dense(1, name='positive')(lstm)
    dea = Dense(1, name='deaths')(lstm)
    r0 = Dense(1, name='r0')(lstm)
    model = Model(inputs=inp, outputs=[pos, dea, r0])
    return model

mse = []
r2 = []
max_e = []
loss = []

callbacks = [tf.keras.callbacks.ModelCheckpoint(f'./checkpoints/{model_config}.h5', save_best_only=True, monitor='val_loss', mode='min'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True)]
# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, restore_best_weights=True)]

kf = KFold(10, random_state=0, shuffle=True)
for foldIdx, (train_index, test_index) in enumerate(kf.split(X)):
    print(f' fold {foldIdx} '.center(80, '-'))
    xTrain, yTrain = X[train_index], Y[train_index]
    xTest, yTest = X[test_index], Y[test_index]

    model = create_model()
    if foldIdx == 0:
        model.summary()
        default_weights = model.get_weights()
    else:
        model.set_weights(default_weights)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=1000, decay_rate=0.9)

    opt = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss={
        'positive': tf.keras.losses.MeanSquaredError(),
        'deaths': tf.keras.losses.MeanSquaredError(),
        'r0': tf.keras.losses.MeanSquaredError()
    }, optimizer=opt)

    history = model.fit(xTrain,
                        {
                            'positive': yTrain[:, 0],
                            'deaths': yTrain[:, 1],
                            'r0': yTrain[:, 2]
                        }, 
                        validation_data=(xTest, {
                            'positive': yTest[:, 0],
                            'deaths': yTest[:, 1],
                            'r0': yTest[:, 2]
                            },
                        ),
                        epochs=1000,
                        batch_size=10,
                        verbose=1,
                        shuffle=True,
                        callbacks=callbacks)

    y_pred = model.predict(xTest)

    y_pred = np.array(y_pred).squeeze(-1).transpose()
    foldMse = list(map(lambda i: mean_squared_error(y_pred[:, i], yTest[:, i]), range(yTest.shape[1])))
    foldR2 = list(map(lambda i: r2_score(y_pred[:, i], yTest[:, i]), range(yTest.shape[1])))
    foldMaxError = list(map(lambda i: max_error(y_pred[:, i], yTest[:, i]), range(yTest.shape[1])))

    loss.append(history.history['val_loss'])
    mse.append(foldMse)
    r2.append(foldR2)
    max_e.append(foldMaxError)

    if mse[-1] == min(mse):
        shutil.copy(f'./checkpoints/{model_config}.h5', f'./checkpoints/{model_config}.best.h5')

mse = np.average(np.array(mse), axis=0)
r2 = np.average(np.array(r2), axis=0)
max_e = np.average(np.array(max_e), axis=0)

metrics = {'mse':list(mse), 'r2':list(r2), 'max_e':list(max_e)}

print('seed:', seed)
print('metrics:', metrics)

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 9))
for i, l in enumerate(loss):
    plt.plot(l, label=f'fold {i}')
plt.grid('on')
plt.legend()
plt.savefig(f'../images/{model_config}.loss.png')
plt.show()

pred = model.predict(X[:X.shape[0]//3])
pred = np.array(pred).squeeze(-1).transpose()
truth = Y[:X.shape[0]//3]

f, ax = plt.subplots(3, 1, figsize=(10, 6))
for i in range(truth.shape[1]):
    ax[i].plot(pred[:,i], c='g')
    ax[i].plot(truth[:, i], c='k')
    ax[i].grid('on')
plt.savefig(f'../images/{model_config}.alldata.png')
plt.show()
