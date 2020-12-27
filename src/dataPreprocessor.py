#from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def preprocessData_jj6ziyrp(df):
    print(df.columns)

    #Remove missing values
    df = df.dropna()

    df = df.drop(['comarcadescripcio', 'sexedescripcio', 'municipidescripcio'], axis=1)
    print(df.columns)

    # list of categorical features
    cols = df.columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = list(set(cols) - set(num_cols))
    print('\ncat cols: ' + str(cat_cols))
    print('num cols: ' + str(num_cols))

    return df

def preprocessData_uqk7bf9s(df):
    #Que fem amb la date???
    print(df.columns)
    #Remove missing values
    df = df.dropna()

    df = df.drop(['comarcadescripcio','sexedescripcio'], axis=1)
    print(df.columns)

    # list of categorical features
    cols = df.columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = list(set(cols) - set(num_cols))
    print('\ncat cols: ' + str(cat_cols))
    print('num cols: ' + str(num_cols))

    #Fer one hot encoding de la columna de tipuscas descripcio

    return df

def preprocessData_xuwfdxjd(df):
    print(df.columns)

    df = df.drop(['regiosanitariadescripcio','sectorsanitaridescripcio', 'absdescripcio', 'sexedescripcio'], axis=1)
    print(df.columns)

    # list of categorical features
    cols = df.columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = list(set(cols) - set(num_cols))
    print('\ncat cols: ' + str(cat_cols))
    print('num cols: ' + str(num_cols))

    return df

def preprocessData(df,id):
    print("*************************** Preprocessing Data "+id+" ***************************")
    if id == 'jj6z-iyrp':
        df = preprocessData_jj6ziyrp(df)
    elif id == 'uqk7-bf9s':
        df = preprocessData_uqk7bf9s(df)
    elif id == 'xuwf-dxjd':
        df = preprocessData_xuwfdxjd(df)
    return df
