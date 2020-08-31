#!/usr/bin/env python

import pandas as pd
import random
import argparse
from sklearn import preprocessing
from os import path


def read_FFT_doc(name_doc):

    matrix_data = []

    file_doc = open(name_doc, "r")

    line = file_doc.readline()

    while line:

        data = line.replace("\n", "").split(",")
        for i in range(len(data)):
            data[i] = float(data[i])
        matrix_data.append(data)
        line = file_doc.readline()

    file_doc.close()
    return matrix_data


def get_response_from_dataset(name_doc):

    response_array = []

    file_doc = open(name_doc, "r")

    line = file_doc.readline()

    while line:

        data = line.replace("\n", "").split(",")

        response_array.append(data[-1])
        line = file_doc.readline()

    file_doc.close()
    return response_array


# TODO rework this script
def main():

    args = parse_arguments()

    number_examples = args.size

    # make randomized array index element
    index_array = [x for x in range(number_examples)]
    random.shuffle(index_array)

    training_len = int(number_examples * 0.8)
    testing_len = int(number_examples * 0.2)

    total = training_len + testing_len
    diff = number_examples - total
    testing_len = testing_len + diff

    print("Process property: ", args.encoding)

    name_doc_response = args.input_1
    name_doc_FFT = args.input_2

    # read data
    matrix_encoding = read_FFT_doc(name_doc_FFT)
    response_data = get_response_from_dataset(name_doc_response)

    # create dataset
    header = []
    for i in range(len(matrix_encoding[0])):
        header.append("P_" + str(i + 1))

    # scale dataset
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_scaler = min_max_scaler.fit_transform(matrix_encoding)

    dataset = pd.DataFrame(dataset_scaler, columns=header)
    dataset["response"] = response_data

    # export dataset
    dataset.to_csv(path.join(args.output, "dataset_full.csv"), index=False)

    matrix_dataset = []
    # for i in range(len(dataset)):
    # 	row = matrix_encoding[i]
    # 	row.append(response_data[i])
    # 	matrix_dataset.append(row)

    for i in range(len(dataset)):
        row = []
        for key in dataset.keys():
            row.append(dataset[key][i])
        matrix_dataset.append(row)

    # create two datasets: training and testing
    matrix_dataset_training = []
    matrix_dataset_testing = []

    # preparing training
    for i in range(training_len):

        row_data = matrix_dataset[index_array[i]]
        matrix_dataset_training.append(row_data)

    # preparing testing
    for i in range(training_len, training_len + testing_len):

        row_data = matrix_dataset[index_array[i]]
        matrix_dataset_testing.append(row_data)

    header.append("response")
    # export dataset
    dataset_testing = pd.DataFrame(matrix_dataset_testing, columns=header)
    dataset_testing.to_csv(path.join(args.output, "testing_dataset.csv"), index=False)

    dataset_training = pd.DataFrame(matrix_dataset_training, columns=header)
    dataset_training.to_csv(path.join(args.output, "training_dataset.csv"), index=False)


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser(
        "Split the encoded dataset into three csv files: 1.'dataset_full.csv'  2.'training_dataset.csv'  3.'testing_dataset.csv'"
    )

    parser.add_argument(
        "-i1",
        "--input-1",
        action="store",
        required=True,
        help="csv file with the encoded dataset with class",
    )

    parser.add_argument(
        "-i2",
        "--input-2",
        action="store",
        required=True,
        help="csv file with the Fourier transform encoded dataset",
    )

    parser.add_argument(
        "-e",
        "--encoding",
        action="store",
        required=True,
        help="encoding used on the input dataset",
    )

    parser.add_argument(
        "-s",
        "--size",
        action="store",
        required=True,
        type=int,
        help="Int number with the number of examples to use from the encoded dataset",
    )

    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        choices=['regression', 'classification'],
        help="Type of dataset {regression|classification}",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="output path for training and testing dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
