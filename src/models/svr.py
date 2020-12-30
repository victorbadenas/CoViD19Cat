from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from utils import cvEvaluateModel
import logging

def findBestSVR(X, Y):
    params = {'C': [1.0,10.0,100.0], 'kernel': ['rbf','linear', 'poly'], 'degree': [2,3,4,5]}

    svr = GridSearchCV(SVR(),
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=1)

    svr = svr.fit(X, Y)
    best_params = svr.best_params_

    finalSVR = SVR(**best_params)

    metrics, finalSVR = cvEvaluateModel(X, Y, finalSVR)

    allprediction = finalSVR.predict(X)

    logging.info(f"best params for SVR: {best_params}")
    logging.info(f"best model with metrics: {metrics}")

    return allprediction[:, 0], allprediction[:, 1], allprediction[:, 2], best_params, metrics