from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def findBestMlp(X, Y, xTrain, yTrain, xTest, yTest):

    hiddenSizes = [(10,), (50,), (100,), (100, 10), (100, 50)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    batch_size = [1, 5, 10, 50]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    params = dict(hidden_layer_sizes=hiddenSizes,
                  activation=activation,
                  batch_size=batch_size,
                  learning_rate=learning_rate)

    mlpcv = GridSearchCV(MLPRegressor(max_iter=1000, tol=1e-6), 
                        params,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        cv=10,
                        refit=False,
                        verbose=2)
    mlpcv = mlpcv.fit(X, Y)
    return mlpcv.best_params_
