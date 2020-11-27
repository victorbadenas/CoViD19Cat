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

def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids', action="append", default=["xuwf-dxjd", "jj6z-iyrp"])
    parser.add_argument('-s', '--show', action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=Path, default="./log/log.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    set_logger(args.log_file, args.debug)
    show_parameters(args)
    main(args)
