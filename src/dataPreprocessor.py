from sklearn.preprocessing import LabelEncoder
import numpy as np


def preprocessData(df, id:str):
    print("*************************** Preprocessing Data "+id+" ***************************")

    df = df.dropna()

    if id == 'xuwf-dxjd':
        df = df.drop(['regiosanitariadescripcio', 'sectorsanitaridescripcio', 'absdescripcio', 'sexedescripcio'], axis=1)
    else:
        if id == 'jj6z-iyrp':
            df = df.drop(['comarcadescripcio', 'sexedescripcio', 'municipidescripcio'], axis=1)
        else:
            df = df.drop(['comarcadescripcio', 'sexedescripcio'], axis=1)

        lb_make = LabelEncoder()
        df['resultatcoviddescripcio'] = lb_make.fit_transform(df['resultatcoviddescripcio'])

    return df
