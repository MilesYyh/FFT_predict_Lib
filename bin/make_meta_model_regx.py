#!/usr/bin/env python

import pandas as pd
import numpy as np
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.metrics import r2_score


def main():

    args = parse_arguments()
    models_list = args.model_results
    response_path = args.response

    # Load models results
    df = pd.concat(
        [pd.read_csv(model_path, sep=",", index_col=0) for model_path in models_list]
    )
    matrix_response = df.to_numpy()

    # Get actual response
    response_original = pd.read_csv(response_path, sep=",", index_col=0)[
        "response"
    ].to_list()

    # get mean response
    response_predict_avg = []

    for i in range(len(matrix_response[0])):
        point = []
        for j in range(len(matrix_response)):
            point.append(matrix_response[j][i])
        response_predict_avg.append(np.mean(point))

    # get performance compare real value v/s predicted value
    pearson_value = pearsonr(response_original, response_predict_avg)
    spearman_value = spearmanr(response_original, response_predict_avg)
    kendalltau_value = kendalltau(response_original, response_predict_avg)
    r2_score_value = r2_score(response_original, response_predict_avg)

    print("Pearson: ", pearson_value)
    print("Spearman: ", spearman_value)
    print("Kendall: ", kendalltau_value)
    print("R2 score: ", r2_score_value)


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser("Script for regression meta model evaluation")

    parser.add_argument(
        "-m",
        "--model-results",
        action="store",
        nargs="+",
        required=True,
        help="csv files with results from models",
    )

    parser.add_argument(
        "-r",
        "--response",
        action="store",
        required=True,
        help="csv with the original response of the model testing dataset",
    )

    parser.add_argument(
        "-e",
        "--encoding",
        action="store",
        help="encoding used on the input dataset",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        help="output",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
