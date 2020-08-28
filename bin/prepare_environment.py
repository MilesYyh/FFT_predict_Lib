#!/usr/bin/env python

import pandas as pd
import sys
import argparse


def main():
    args = parse_arguments()

    # read dataset and remove outliers
    print("Read csv file")
    dataset = pd.read_csv(args.input)
    path_outout = sys.argv[2]

    print("Remove outliers")
    dataset_filter = dataset.dropna()
    dataset_filter.to_csv(args.output, index=False)

    """
    print("Prepare paths")
    
    # create directory with different properties to use
    list_propertyes = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

    for property_value in list_propertyes:
        command = "mkdir -p %s%s" % (path_outout, property_value)
        print(command)
        os.system(command)
    print("OK-Process")
    """


def parse_arguments():
    """
    Prepare dataset for pipeline

    @return: arguments parser
    """

    parser = argparse.ArgumentParser(
        "Prepare dataset for pipeline"
    )

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="csv file with the dataset, the file should have two columns 'sequence' and 'response'",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="output path for prepared file",
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
