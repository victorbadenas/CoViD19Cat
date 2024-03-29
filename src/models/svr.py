from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from utils import cvEvaluateModel
import logging


def findBestSVR(X, Y):
    predicted_arrays = []
    metrics = []
    for i in range(Y.shape[1]):
        logging.info(f'computing best model for {i}th dimension')
        predicted, best_params, metric = findBestSVRForGivenOutput(X, Y[:, i])
        metrics.append(metric)
        predicted_arrays.append(predicted)

    joinedMetrics = metrics[0]
    for i in range(1, len(metrics)):
        joinedMetrics['mse'].append(metrics[i]['mse'][0])
        joinedMetrics['r2'].append(metrics[i]['r2'][0])
        joinedMetrics['max_e'].append(metrics[i]['max_e'][0])

    return (*predicted_arrays, joinedMetrics)

def findBestSVRForGivenOutput(X, Y):
    params = {'C': [1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'gamma': ['scale','auto'] ,'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'degree': [1, 2, 3, 4, 5, 10]}
    svr = GridSearchCV(SVR(),
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=1)
    svr = svr.fit(X, Y)
    best_params = svr.best_params_

    finalSvr = SVR(**best_params)

    metrics, finalSvr = cvEvaluateModel(X, Y, finalSvr)

    allprediction = finalSvr.predict(X)

    logging.info(f"best params for SVR: {best_params}")
    logging.info(f"best model with metrics: {metrics}")

    return allprediction, best_params, metrics

