import argparse
import pandas as pd
from DataRetriever import DataRetriever
import matplotlib.pyplot as plt

def main(args):
    data = DataRetriever(args)()
    print(data.head())
    data = data.apply(pd.to_numeric)
    data = data.reindex(index=data.index[::-1])
    fig, axes = plt.subplots(figsize=(15, 9), nrows=len(data.columns)//2, ncols=2)
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i//2, i%2])
        axes[i//2, i%2].grid('on')
        axes[i//2, i%2].set_ylabel(column)
    plt.tight_layout()
    plt.show()

def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    main(args)