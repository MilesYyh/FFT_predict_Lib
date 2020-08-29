#!/usr/bin/env python

import pandas as pd
import argparse
from os import path


# function to encoding
def encoding_pca_data(sequence, data_property):
    residues = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    sequence_encoding = []

    for element in sequence:
        # get the pos of residue
        pos = -1
        for i in range(len(residues)):
            if element == residues[i]:
                pos = i
                break
        if pos != -1:
            sequence_encoding.append(data_property["component_1"][pos])

    return sequence_encoding


def main():
    args = parse_arguments()

    dataset = pd.read_csv(args.input)

    # check zero-padding conformation
    two_base_points = []
    for i in range(15):
        two_base_points.append(pow(2, i))

    print("Process property: ", args.encoding)
    property_dataset = pd.read_csv(args.encoding)

    dataset_encoding = []
    length_sequence = []

    for i in range(len(dataset)):
        row_data = []
        # get sequence encoding with PCA Analysis
        sequence_encoding = encoding_pca_data(
            dataset["sequence"][i], property_dataset
        )
        row_data.append(sequence_encoding)
        row_data.append(dataset["response"][i])

        dataset_encoding.append(row_data)
        length_sequence.append(len(row_data[0]))

    # make zero padding
    max_length = max(length_sequence)

    # get value near from two_base_points
    pos_pow = 0
    for i in range(len(two_base_points)):
        dif_data = two_base_points[i] - max_length
        if dif_data >= 0:
            pos_pow = i
            break
    for i in range(len(dataset_encoding)):

        for j in range(len(dataset_encoding[i][0]), two_base_points[pos_pow]):
            dataset_encoding[i][0].append(0)

    # export dataset to csv
    matrix_export = []
    matrix_export_not_class = []

    for element in dataset_encoding:
        row_full = []
        row_normal = []
        for point in element[0]:
            row_full.append(point)
            row_normal.append(point)
        matrix_export_not_class.append(row_normal)
        row_full.append(element[-1])
        matrix_export.append(row_full)

    df_export = pd.DataFrame(matrix_export)
    df_export_not_class = pd.DataFrame(matrix_export_not_class)
    df_export.to_csv(
        path.join(args.output, "encoding_with_class.csv"),
        index=False,
        header=False,
    )
    df_export_not_class.to_csv(
        path.join(args.output, "encoding_without_class.csv"),
        index=False,
        header=False,
    )


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser("Encode dataset file using the 8 meta-properties and produce two files "
                                     "'encoding_with_class.csv' and 'encoding_without_class.csv'")

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="csv file with the dataset, the file should have two columns 'sequence' and 'response'",
    )

    parser.add_argument(
        "-e",
        "--encoding",
        action="store",
        required=True,
        help="encoding csv file that will be applied to the dataset",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="output path for encoded files",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
