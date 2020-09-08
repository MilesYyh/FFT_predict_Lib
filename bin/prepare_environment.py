#!/usr/bin/env python

import pandas as pd
import sys
import argparse
from sklearn import preprocessing


def main():
    args = parse_arguments()

    # read dataset and remove outliers
    print("Read csv file")
    dataset = pd.read_csv(args.input)

    print("Remove outliers")
    dataset_filter = dataset.dropna()

    # Encode string labels to int
    if args.mode == "classification":
        # Transform string labels to int
        le = preprocessing.LabelEncoder()
        le.fit(dataset["response"])
        response_encoded = le.transform(dataset["response"])
        dataset["response"] = response_encoded

    elif args.mode == "regression":
        # Maybe add something here later
        pass

    dataset_filter.to_csv(args.output, index=False)


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser("Prepare dataset for pipeline")

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="csv file with the dataset, the file should have two columns 'sequence' and 'response'",
    )

    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        required=True,
        choices=["regression", "classification"],
        help="Type of dataset {regression|classification}",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="output path for prepared file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
