import os
import sys
import argparse
import pandas as pd
import datetime
from pathlib import Path
from sodapy import Socrata

class DataRetriever:

    MAX_TRIES = 3
    SOCRATA_LIMIT = int(1e6)

    def __init__(self, dataId:str):
        self.client = Socrata("analisi.transparenciacatalunya.cat", None)
        self.dataId = dataId
        self.savePath = self.createSavePath()

    def __call__(self):
        if self.savePath.exists():
            data = self.loadPreviousCsv()
        else:
            data =  self.retrieveData(self.dataId)
            self.saveData(data)
        return data

    def retrieveData(self, dataId):
        data = pd.DataFrame.from_records(self.client.get(dataId, limit=self.SOCRATA_LIMIT))
        limit = self.SOCRATA_LIMIT
        for _ in range(self.MAX_TRIES):
            # data retrieved has been truncated, agument limit
            limit *= 10
            data = pd.DataFrame.from_records(self.client.get(dataId, limit=limit))
            if len(data) != limit:
                break
        else:
            raise ValueError(f"full data has not been retrieved. Only {len(data)} lines")

        data['date'] = data['data']
        data = data.drop('data', axis=1)
        data['date'] = data['date'].apply(lambda x: x.split('T')[0].replace('2020-', ''))
        data = data.set_index('date')
        return data

    def saveData(self, data):
        data.to_csv(self.savePath)

    def loadPreviousCsv(self):
        return pd.read_csv(self.savePath, index_col=0)

    def createSavePath(self):
        date = datetime.datetime.now()
        dataPath = Path(f"data/{self.dataId}/")
        dataPath.mkdir(parents=True, exist_ok=True)
        return dataPath / f"{date.strftime('%Y-%m-%d')}.csv"

def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    DataRetriever(args)()