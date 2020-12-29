from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from itertools import product
import logging

def findBestMlp(X, Y, xTrain, xTest, yTrain, yTest):

    hiddenSizes = list(product([10, 25, 100], [10, 25, 100]))
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

    finalMlp = MLPRegressor(max_iter=1000, tol=1e-6, batch_size=10, **best_params).fit(xTrain, yTrain)
    allprediction = finalMlp.predict(X)
    logging.info(f"best params for MLP: {best_params}")
    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params
