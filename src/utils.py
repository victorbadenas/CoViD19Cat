import sys
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, max_error
import numpy as np

def set_logger(log_file_path, debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging_format = '[%(asctime)s][%(filename)s(%(lineno)d):%(funcName)s]-%(levelname)s: %(message)s'
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=level, format=logging_format)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(consoleHandler)

def show_parameters(parameters):
    logging.info("Called with parameters:")
    for label, value in parameters.__dict__.items():
        logging.info(f"\t{label}: {value}")

def cvEvaluateModel(X, Y, model, return_last_fitted=True):
    assert hasattr(model, 'fit'), 'model does not have a fit method'
    assert hasattr(model, 'predict'), 'model does not have a predict method'

    mse = []
    r2 = []
    max_e = []

    kf = KFold(10, random_state=0, shuffle=True)
    for train_index, test_index in kf.split(X): 
        xTrain, yTrain = X[train_index], Y[train_index]
        xTest, yTest = X[test_index], Y[test_index]

        model.fit(xTrain, yTrain)
        foldPrediction = model.predict(xTest)
        
        foldMse = list(map(lambda i: mean_squared_error(foldPrediction[:, i], yTest[:, i]), range(yTest.shape[1])))
        foldR2 = list(map(lambda i: r2_score(foldPrediction[:, i], yTest[:, i]), range(yTest.shape[1])))
        foldMaxError = list(map(lambda i: max_error(foldPrediction[:, i], yTest[:, i]), range(yTest.shape[1])))

        mse.append(foldMse)
        r2.append(foldR2)
        max_e.append(foldMaxError)

    mse = np.average(np.array(mse), axis=0)
    r2 = np.average(np.array(r2), axis=0)
    max_e = np.average(np.array(max_e), axis=0)

    metrics = {'mse':list(mse), 'r2':list(r2), 'max_e':list(max_e)}
    return metrics, model if return_last_fitted else metrics