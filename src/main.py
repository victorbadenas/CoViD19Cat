import argparse
from pathlib import Path
from dataRetriever import DataRetriever
from utils import set_logger, show_parameters
from dataPreprocessor import preprocessData
from dataPreprocessor import normalizeData
from sklearn.model_selection import train_test_split
from models.mlp import findBestMlp

def main(args):
    prep_data = []
    print("********************   Downloading data from ID: "+args.ids[0]+"   ********************")
    data_pos = DataRetriever(args.ids[0])()
    data_death = DataRetriever(args.ids[1])()
    dataset = preprocessData(data_pos, data_death)
    data, dates = normalizeData(dataset)

    X = data[:-1]
    Y = data[:,[0,1,-1]][1:]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=.1)
    print(xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)

    # findBestMlp(X, Y, xTrain, xTest, yTrain, yTest)
    return


def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids', action="append", default=["jj6z-iyrp", "uqk7-bf9s"])
    parser.add_argument('-s', '--show', action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=Path, default="./log/log.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    set_logger(args.log_file, args.debug)
    show_parameters(args)
    main(args)
