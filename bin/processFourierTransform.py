#!/usr/bin/env python
import pandas as pd
from scipy.fft import fft
import numpy as np
import argparse


def main():
    args = parse_arguments()

    # inputs
    dataset_to_process = pd.read_csv(args.input)
    output_spectra = args.spectral_output
    output_domain = args.domain_output

    # process matrix data
    columns = ["P_{}".format(i) for i in range(len(dataset_to_process.columns))]
    dataset_to_process.columns = columns

    matrix_encoding = []
    domain_data = []

    for i in range(len(dataset_to_process)):

        # get a sequences
        sequence_encoding = [dataset_to_process[key][i] for key in dataset_to_process.keys()]
        number_sample = len(sequence_encoding)

        # sample spacing
        T = 1.0/float(number_sample)
        x = np.linspace(0.0, number_sample*T, number_sample)
        yf = fft(sequence_encoding)
        xf = np.linspace(0.0, 1.0/(2.0*T), number_sample//2)
        matrix_encoding.append(np.abs(yf[0:number_sample//2]))
        domain_data.append(xf)

    encoding_dataset = pd.DataFrame(matrix_encoding)
    domain_dataset = pd.DataFrame(domain_data)

    encoding_dataset.to_csv(output_spectra, index=False, header=False)
    domain_dataset.to_csv(output_domain, index=False, header=False)


def parse_arguments():

    parser = argparse.ArgumentParser(
        "Apply fast fourier transform"
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--spectral-output",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--domain-output",
        action="store",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
