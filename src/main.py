import argparse
import logging
import math
import pandas as pd
from pathlib import Path
from dataRetriever import DataRetriever
import matplotlib.pyplot as plt
from utils import set_logger, show_parameters

def main(args):
    for dataId in args.ids:
        data = DataRetriever(dataId)()
    data = DataRetriever("623z-r97q")()
    data = data.apply(pd.to_numeric)
    data = data.reindex(index=data.index[::-1])
    fig, axes = plt.subplots(figsize=(15, 9), nrows=int(math.ceil(len(data.columns)/2)), ncols=2)
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i//2, i%2])
        axes[i//2, i%2].grid('on')
        axes[i//2, i%2].set_ylabel(column)
    plt.tight_layout()
    if args.show:
        plt.show()

def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids', action="append", default=["623z-r97q", "jj6z-iyrp"])
    parser.add_argument('-s', '--show', action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=Path, default="./log/train.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    set_logger(args.log_file, args.debug)
    show_parameters(args)
    main(args)
