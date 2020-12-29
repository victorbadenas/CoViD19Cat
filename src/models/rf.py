import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import logging


def findBestRF(X, Y, xTrain, xTest, yTrain, yTest):
    params = {'n_estimators': [150, 200, 500]}
    rfcv = GridSearchCV(RandomForestRegressor(), 
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=1)
    rfcv = rfcv.fit(X, Y)
    best_params = rfcv.best_params_

    finalRF = RandomForestRegressor(**best_params).fit(xTrain, yTrain)
    allprediction = finalRF.predict(X)
    logging.info(f"best params for RF: {best_params}")
    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params
