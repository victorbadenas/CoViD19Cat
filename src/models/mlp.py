from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from itertools import product
from utils import cvEvaluateModel
import logging
import numpy as np

def findBestMlp(X, Y):

    hiddenSizes = list(product([5, 8, 10], [5, 8, 10]))
    hiddenSizes += [(5,), (8,), (10,)]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    params = dict(hidden_layer_sizes=hiddenSizes,
                  learning_rate=learning_rate)

    mlpcv = GridSearchCV(MLPRegressor(max_iter=1000, tol=1e-6, batch_size=10), 
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=2)
    mlpcv = mlpcv.fit(X, Y)
    best_params = mlpcv.best_params_

    finalMlp = MLPRegressor(max_iter=1000, tol=1e-6, batch_size=10, **best_params)

    metrics, finalMlp = cvEvaluateModel(X, Y, finalMlp)

    allprediction = finalMlp.predict(X)

    logging.info(f"best params for MLP: {best_params}")
    logging.info(f"best model with metrics: {metrics}")

    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params, metrics
