import numpy as np

def augmentData(X:np.ndarray, Y:np.ndarray):
    assert X.shape[0] == Y.shape[0], "arrays are inconsistant. axis 0 must be of the same size"

    Xextended = X.copy()
    Yextended = Y.copy()
    for std in [1e-4, 5e-5]:
        posRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,))
        deaRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,))
        r0Random = np.random.normal(loc=0, scale=std, size=(X.shape[0]+1,))
        otherRandom = np.random.normal(loc=0, scale=std, size=(X.shape[0],X.shape[1]-3))

        Xrand = np.concatenate([posRandom[:-1, None], deaRandom[:-1, None], otherRandom, r0Random[:-1, None]], axis=1)
        Xrand += X
        Yrand = np.concatenate([posRandom[1:, None], deaRandom[1:, None], r0Random[1:, None]], axis=1)
        Yrand += Y
        Xextended = np.concatenate((Xextended, Xrand))
        Yextended = np.concatenate((Yextended, Yrand))
    return Xextended, Yextended
