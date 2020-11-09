import os
import sys
import argparse
import pandas as pd
import datetime
from pathlib import Path
from sodapy import Socrata

class DataRetriever:
    def __init__(self, args):
        self.args = args
        self.client = Socrata("analisi.transparenciacatalunya.cat", None)
        self.dataId = "623z-r97q"
        self.savePath = self.createSavePath()

    def __call__(self):
        if self.savePath.exists():
            data = self.loadPreviousCsv()
        else:
            data =  self.retrieveData(self.dataId)
            self.saveData(data)
        return data.set_index('date')

    def retrieveData(self, dataId):
        data = pd.DataFrame.from_records(self.client.get(dataId, limit=2000))
        data['date'] = data['data']
        data = data.drop('data', axis=1)
        data['date'] = data['date'].apply(lambda x: x.split('T')[0].replace('2020-', ''))
        return data

    def saveData(self, data):
        data.to_csv(self.savePath)

    def loadPreviousCsv(self):
        return pd.read_csv(self.savePath)

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