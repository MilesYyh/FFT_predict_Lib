#!/usr/bin/env python

import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy.random import RandomState
from os import path

TEST_SIZE = 0.2
SAMPLE_SEED = 51
SPLIT_SEED = 42


def main():

    args = parse_arguments()

    sample_fraction = args.size
    encode_dataset = args.input_1
    fft_file = args.input_2

    # Load data
    matrix_encoding = pd.read_csv(fft_file, sep=',', header=None)
    min_max_scaler = preprocessing.MinMaxScaler()
    matrix_encoding_scaled = min_max_scaler.fit_transform(matrix_encoding)

    dataset = pd.DataFrame(matrix_encoding_scaled)
    dataset.columns = [f"P_{i+1}" for i in range(len(dataset.columns))]

    dataset['response'] = pd.read_csv(encode_dataset, header=None, sep=',').iloc[:, -1]

    # Save full dataset
    dataset.to_csv(path.join(args.output, "dataset_full.csv"), index=True, float_format='%.5f')

    # Random sample the dataset
    sample_dataset = dataset.sample(frac=sample_fraction, axis=0, random_state=RandomState(SAMPLE_SEED))
    labels = sample_dataset['response']
    data = sample_dataset.drop(['response'], axis=1)

    # Split random sample into testing and training datasets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, random_state=RandomState(SPLIT_SEED))
    training_df = pd.concat([x_train, y_train], axis=1)
    testing_df = pd.concat([x_test, y_test], axis=1)

    # Save datasets
    training_df.to_csv(path.join(args.output, "training_dataset.csv"), index=True, float_format='%.5f')
    testing_df.to_csv(path.join(args.output, "testing_dataset.csv"), index=True, float_format='%.5f')


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
        "-s",
        "--size",
        action="store",
        required=True,
        type=float,
        help="float number between [0.0-1.0] with the fraction of examples to use from the encoded dataset",
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
