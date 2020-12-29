import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging


def findBestRF(X, Y, xTrain, xTest, yTrain, yTest):
    '''
    hiddenSizes = [(10,), (50,), (100,), (100, 10), (100, 50)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    batch_size = [1, 5, 10, 50]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    params = dict(hidden_layer_sizes=hiddenSizes,
                  activation=activation,
                  batch_size=batch_size,
                  learning_rate=learning_rate)
    '''
    params = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
        }
    rfcv = GridSearchCV(RandomForestRegressor(), 
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=2)
    rfcv = rfcv.fit(X, Y)
    best_params = rfcv.best_params_

    finalRF = RandomForestRegressor(**best_params).fit(xTrain, yTrain)
    allprediction = finalRF.predict(X)
    logging.info(f"best params for RF: {best_params}")
    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.1)
    print(xTrain.shape, yTrain.shape)
    findBestRF(X, y, xTrain, xTest, yTrain, yTest)
