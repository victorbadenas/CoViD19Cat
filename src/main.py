import argparse
import logging
from pathlib import Path
from src.dataRetriever import DataRetriever
from src.utils import set_logger, show_parameters
from src.dataPreprocessor import preprocessData, normalizeData
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.models import findBestSVR, findBestRF, findBestMlp, findBestAda

def main(args):

    logging.info(f"Downloading data from ID: {args.ids[0]}".center(80, '*'))
    data_pos = DataRetriever(args.ids[0])()
    data_death = DataRetriever(args.ids[1])()
    dataset = preprocessData(data_pos, data_death)
    data, dates = normalizeData(dataset)

    X = data[:-1]
    Y = data[:,[0,1,-1]][1:]


    infectedSvrPred, deathsSvrPred, r0SvrPred, svrMetrics = findBestSVR(X, Y)
    infectedAdaPred, deathsAdaPred, r0AdaPred, adaMetrics = findBestAda(X, Y)
    infectedRfPred, deathsRfPred, r0RfPred, bestRfParams, rfMetrics = findBestRF(X, Y)
    infectedMlpPred, deathsMlpPred, r0MlpPred, bestMlpParams, mlpMetrics = findBestMlp(X, Y)



    f, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax[0].plot(infectedRfPred, c='r', label='rf_predicted')
    ax[0].plot(infectedMlpPred, c='g', label='mlp_predicted')
    ax[0].plot(infectedAdaPred, c='k', label='ada_predicted')
    ax[0].plot(infectedSvrPred, c='m', label='svr_predicted')
    ax[0].plot(Y[:, 0], c='b', label='truth')
    ax[0].set_ylabel('num_infected')
    ax[0].grid('on')
    ax[0].legend()

    ax[1].plot(deathsRfPred, c='r', label='rf_predicted')
    ax[1].plot(deathsMlpPred, c='g', label='mlp_predicted')
    ax[1].plot(deathsAdaPred, c='k', label='ada_predicted')
    ax[1].plot(deathsSvrPred, c='m', label='svr_predicted')
    ax[1].plot(Y[:, 1], c='b', label='truth')
    ax[1].set_ylabel('num_deaths')
    ax[1].grid('on')
    ax[1].legend()

    ax[2].plot(r0RfPred, c='r', label='rf_predicted')
    ax[2].plot(r0MlpPred, c='g', label='mlp_predicted')
    ax[2].plot(r0AdaPred, c='k', label='ada_predicted')
    ax[2].plot(r0SvrPred, c='m', label='ada_predicted')
    ax[2].plot(Y[:, 2], c='b', label='truth')
    ax[2].set_ylabel('R0')
    ax[2].grid('on')
    ax[2].legend()

    plt.xticks(range(0, len(dates[1:]), 7), dates[1::7], rotation=90)
    plt.show()


def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids', action="append", default=["jj6z-iyrp", "uqk7-bf9s"])
    #parser.add_argument('-s', '--show', action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=Path, default="./log/log.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    set_logger(args.log_file, args.debug)
    show_parameters(args)
    main(args)
