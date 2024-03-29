import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from utils import cvEvaluateModel
import logging


def findBestRF(X, Y):
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

    finalRF = RandomForestRegressor(**best_params)

    metrics, finalRF = cvEvaluateModel(X, Y, finalRF)

    allprediction = finalRF.predict(X)

    logging.info(f"best params for RF: {best_params}")
    logging.info(f"best model with metrics: {metrics}")

    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params, metrics
