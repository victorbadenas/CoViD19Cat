from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from utils import cvEvaluateModel
import logging

def findBestAda(X, Y):
    predicted_arrays = []
    metrics = []
    for i in range(Y.shape[1]):
        logging.info(f'computing best model for {i}th dimension')
        predicted, best_params, metric = findBestAdaForGivenOutput(X, Y[:, i])
        metrics.append(metric)
        predicted_arrays.append(predicted)

    joinedMetrics = metrics[0]
    for i in range(1, len(metrics)):
        joinedMetrics['mse'].append(metrics[i]['mse'][0])
        joinedMetrics['r2'].append(metrics[i]['r2'][0])
        joinedMetrics['max_e'].append(metrics[i]['max_e'][0])

    return (*predicted_arrays, joinedMetrics)

def findBestAdaForGivenOutput(X, Y):
    n_estimators = [150, 200, 500]
    learning_rate = [1, .9, .8, .5]
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate)
    adacv = GridSearchCV(AdaBoostRegressor(), 
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=2)
    adacv = adacv.fit(X, Y)
    best_params = adacv.best_params_

    finalAda = AdaBoostRegressor(**best_params)

    metrics, finalAda = cvEvaluateModel(X, Y, finalAda)

    allprediction = finalAda.predict(X)

    logging.info(f"best params for Adaboost: {best_params}")
    logging.info(f"best model with metrics: {metrics}")

    return allprediction, best_params, metrics
