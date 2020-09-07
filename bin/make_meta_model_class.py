#!/usr/bin/env python

import sys
from joblib import dump, load
from sklearn import preprocessing
import glob
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def main():
    args = parse_arguments()
    models_list = args.model_results
    response_path = args.response

    # Load models results
    df = pd.concat([pd.read_csv(model_path, sep=',', index_col=0) for model_path in models_list])
    matrix_response = df.to_numpy()

    # Get actual response
    response_original = pd.read_csv(response_path, sep=',', index_col=0)['response'].to_list()

    # get mean response
    response_predict_voted = []

    for i in range(len(matrix_response[0])):
        point = []
        for j in range(len(matrix_response)):
            point.append(matrix_response[j][i])

        unique_responses = list(set(point))

        # count for each response
        counts_data = {}
        counts_array = []
        for response in unique_responses:
            cont = 0
            for element in point:
                if response == element:
                    cont += 1
            counts_data.update({str(response): cont})
            counts_array.append(cont)

        max_cont = max(counts_array)

        response = -1
        for key in counts_data:
            if counts_data[key] == max_cont:
                response = int(key)
                break

        response_predict_voted.append(response)

    # get performance compare real value v/s predicted value
    accuracy_value = accuracy_score(response_original, response_predict_voted)
    f1_value = f1_score(response_original, response_predict_voted, average='weighted')
    precision_value = precision_score(response_original, response_predict_voted, average='weighted')
    recall_value = recall_score(response_original, response_predict_voted, average='weighted')

    print("Accuracy: ", accuracy_value)
    print("Recall: ", recall_value)
    print("Precision: ", precision_value)
    print("F1 score: ", f1_value)


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser(
        "Script for classification meta model evaluation"
    )

    parser.add_argument(
        "-m",
        "--model-results",
        action="store",
        nargs='+',
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


if __name__ == '__main__':
    main()
