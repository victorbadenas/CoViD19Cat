import numpy as np
import unidecode
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

def createPivotTable(data, newdf, id, tag):
    columns = []

    if id == 0 : columns = ['date', 'numcasos', 'comarcadescripcio']
    if id == 2: columns = ['sexedescripcio', 'date', 'numcasos']
    if id == 1: columns = ['date', 'numexitus', 'comarcadescripcio']
    if id == 3: columns = ['sexedescripcio', 'date', 'numexitus']

    databyRegion = data[columns].dropna()

    if id == 0 or id == 1:
        databyRegion = databyRegion.pivot_table(index=['date'], columns='comarcadescripcio', aggfunc=sum).fillna(0.0)
    if id > 1:
        databyRegion = databyRegion.pivot_table(index=['date'], columns='sexedescripcio',aggfunc=sum).fillna(0.0)
    if id == 0 or id == 2:
        databyRegion = databyRegion['numcasos']
    if id == 1 or id == 3:
        databyRegion = databyRegion['numexitus']

    databyRegion.columns = [unidecode.unidecode(columnName.replace(' ', '')) + tag for columnName in databyRegion.columns]
    newdf = newdf.join(databyRegion).fillna(0.0)

    return newdf

def addRo(rodf, newdf):

    LOOKBACK = 7
    POPULATION = 7.566e7
    array = rodf.to_numpy().squeeze(-1)
    size = len(array)
    R0 = np.zeros(size)

    for i in range(LOOKBACK, len(array)):
        if array[i - LOOKBACK] == 0:
            R0[i] = 1
        else:
            R0[i] = np.exp((((np.log(POPULATION / ((1 / (array[i - 1] / (array[i - LOOKBACK] * POPULATION))) - 1))) / (
                        LOOKBACK - 1))))
    R0[np.isnan(R0)] = 0.0
    R0[np.isinf(R0)] = 0.0
    rodf['R0'] = R0
    rodf = rodf.drop('numcasos', axis=1)
    newdf = newdf.join(rodf).fillna(0.0)

    return newdf

def preprocessData(infectionData, deathData):
    print("*************************** Preprocessing Data COVID-19 ***************************")

    infectionData['numcasos'][infectionData['resultatcoviddescripcio'] == 'Sospitós'] = 0.0
    infections = infectionData[['date', 'numcasos']].set_index('date').groupby('date').sum()
    deaths = deathData[['date', 'numexitus']].set_index('date').groupby('date').sum()
    newdf = infections.join(deaths).fillna(0.0)

    newdf = createPivotTable(infectionData, newdf, 0, '_pos')
    newdf = createPivotTable(deathData, newdf, 1, '_death')
    newdf = createPivotTable(infectionData, newdf, 2, '_pos')
    newdf = createPivotTable(deathData, newdf, 3, '_death')

    newdf = addRo(infections.copy(), newdf)

    return newdf

def normalizeData(newdf):
    positiveScaler = MinMaxScaler(feature_range=(0, .5))
    deathsScaler = MinMaxScaler(feature_range=(0, .5))
    positiveHistScaler = MinMaxScaler(feature_range=(0, .5)).fit(newdf.to_numpy()[:, 2:44].flatten()[:, None])
    deathsHistScaler = MinMaxScaler(feature_range=(0, .5)).fit(newdf.to_numpy()[:, 44:87].flatten()[:, None])
    positiveSexHistScaler = MinMaxScaler(feature_range=(0, .5)).fit(newdf.to_numpy()[:, 87:89].flatten()[:, None])
    deathsSexHistScaler = MinMaxScaler(feature_range=(0, .5)).fit(newdf.to_numpy()[:, 89:91].flatten()[:, None])
    r0Scaler = MinMaxScaler(feature_range=(0, .5))

    newdf['numcasos'] = positiveScaler.fit_transform(newdf['numcasos'].to_numpy().reshape(-1, 1))
    newdf['numexitus'] = deathsScaler.fit_transform(newdf['numexitus'].to_numpy().reshape(-1, 1))
    newdf['R0'] = r0Scaler.fit_transform(newdf['R0'].to_numpy().reshape(-1, 1))
    for column in newdf.columns[2:44]:
        newdf[column] = positiveHistScaler.transform(newdf[column].to_numpy().reshape(-1, 1))
    for column in newdf.columns[44:87]:
        newdf[column] = deathsHistScaler.transform(newdf[column].to_numpy().reshape(-1, 1))
    for column in newdf.columns[87:89]:
        newdf[column] = positiveSexHistScaler.transform(newdf[column].to_numpy().reshape(-1, 1))
    for column in newdf.columns[89:91]:
        newdf[column] = deathsSexHistScaler.transform(newdf[column].to_numpy().reshape(-1, 1))

    data = newdf.to_numpy()
    dates = newdf.index.to_list()

    return data, dates