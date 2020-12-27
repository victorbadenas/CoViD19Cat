import argparse
from pathlib import Path
from src.dataRetriever import DataRetriever
from src.utils import set_logger, show_parameters
from src.dataPreprocessor import preprocessData

def main(args):
    prep_data = []
    for dataId in args.ids:
        data = DataRetriever(dataId)()
        prep_data.append(preprocessData(data, dataId))


def ParseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ids', action="append", default=["xuwf-dxjd", "jj6z-iyrp", "uqk7-bf9s"])
    parser.add_argument('-s', '--show', action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("--log_file", type=Path, default="./log/log.log")
    return parser.parse_args()

if __name__ == "__main__":
    args = ParseArgumentsFromCommandLine()
    set_logger(args.log_file, args.debug)
    show_parameters(args)
    main(args)
