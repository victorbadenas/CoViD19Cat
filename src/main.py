import argparse
import logging
from pathlib import Path
from dataRetriever import DataRetriever
from utils import set_logger, show_parameters
from dataPreprocessor import preprocessData, customNormalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import findBestSVR, findBestRF, findBestMlp, findBestAda
from augmentData import augmentData

DATA_AUGMENT_RATIO = 3

def main(args):

    logging.info(f"Downloading data from ID: {args.ids[0]}".center(80, '*'))
    data_pos = DataRetriever(args.ids[0])()
    data_death = DataRetriever(args.ids[1])()
    dataset = preprocessData(data_pos, data_death)
    normalizer = customNormalizer()
    data, dates = normalizer.normalizeData(dataset)

    X = data[:-1]
    Y = data[:,[0,1,-1]][1:]

    X, Y = augmentData(X, Y)

    infectedSvrPred, deathsSvrPred, r0SvrPred, svrMetrics = findBestSVR(X, Y)
    infectedAdaPred, deathsAdaPred, r0AdaPred, adaMetrics = findBestAda(X, Y)
    infectedRfPred, deathsRfPred, r0RfPred, bestRfParams, rfMetrics = findBestRF(X, Y)
    infectedMlpPred, deathsMlpPred, r0MlpPred, bestMlpParams, mlpMetrics = findBestMlp(X, Y)

    final_sample = X.shape[0]//DATA_AUGMENT_RATIO

    # infectedSvrPred, deathsSvrPred, r0SvrPred = normalizer.inverse_transform(infectedSvrPred, deathsSvrPred, r0SvrPred)
    # infectedAdaPred, deathsAdaPred, r0AdaPred = normalizer.inverse_transform(infectedAdaPred, deathsAdaPred, r0AdaPred)
    # infectedRfPred, deathsRfPred, r0RfPred = normalizer.inverse_transform(infectedRfPred, deathsRfPred, r0RfPred)
    # infectedMlpPred, deathsMlpPred, r0MlpPred = normalizer.inverse_transform(infectedSvrPred, deathsSvrPred, r0SvrPred)
    # positiveTruth, deathsTruth, r0Truth = normalizer.inverse_transform(Y[:, 0], Y[:, 1], Y[:, 2])
    positiveTruth, deathsTruth, r0Truth = Y[:, 0], Y[:, 1], Y[:, 2]

    f, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 7))

    ax[0].plot(infectedRfPred[:final_sample], c='r', label='rf_predicted')
    ax[0].plot(infectedMlpPred[:final_sample], c='g', label='mlp_predicted')
    ax[0].plot(infectedAdaPred[:final_sample], c='k', label='ada_predicted')
    ax[0].plot(infectedSvrPred[:final_sample], c='m', label='svr_predicted')
    ax[0].plot(positiveTruth[:final_sample], c='b', label='truth')
    ax[0].set_ylabel('num_infected')
    ax[0].grid('on')
    ax[0].legend()

    ax[1].plot(deathsRfPred[:final_sample], c='r', label='rf_predicted')
    ax[1].plot(deathsMlpPred[:final_sample], c='g', label='mlp_predicted')
    ax[1].plot(deathsAdaPred[:final_sample], c='k', label='ada_predicted')
    ax[1].plot(deathsSvrPred[:final_sample], c='m', label='svr_predicted')
    ax[1].plot(deathsTruth[:final_sample], c='b', label='truth')
    ax[1].set_ylabel('num_deaths')
    ax[1].grid('on')
    ax[1].legend()

    ax[2].plot(r0RfPred[:final_sample], c='r', label='rf_predicted')
    ax[2].plot(r0MlpPred[:final_sample], c='g', label='mlp_predicted')
    ax[2].plot(r0AdaPred[:final_sample], c='k', label='ada_predicted')
    ax[2].plot(r0SvrPred[:final_sample], c='m', label='ada_predicted')
    ax[2].plot(r0Truth[:final_sample], c='b', label='truth')
    ax[2].set_ylabel('R0')
    ax[2].grid('on')
    ax[2].legend()

    plt.xticks(range(0, len(dates[1:]), 7), dates[1::7], rotation=90)
    plt.savefig('images/allpredictions.png')
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
