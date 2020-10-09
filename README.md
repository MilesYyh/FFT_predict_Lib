## README

Predicting the effect of mutations in proteins is one of the most critical challenges in protein engineering; by knowing the effect a substitution of one (or several) residues in the protein's sequence has on its overall properties, could design a variant with a desirable function. New strategies and methodologies to create predictive models are continually being developed. However, those that claim to be general often do not reach adequate performance, and those that aim to a particular task improve their predictive performance at the cost of the method's generality. Moreover, these approaches typically require a particular decision to encode the amino acidic sequence, without an explicit methodological agreement in such endeavor. To address this issues, in this work, we applied clustering, embedding, and dimensionality reduction techniques to the AAIndex database to select meaningful combinations of physicochemical properties for the encoding stage. We then used the chosen set of properties to obtain several encodings of the same sequence, to subsequently apply the Fast Fourier Transform (FFT) on them. We perform an exploratory stage of Machine-Learning models in the frequency space, using different algorithms and hyperparameters. Finally, we select the best performing predictive models in each set of properties and create a synthetic assembled model. We extensively tested the proposed methodology on different datasets and demonstrated that the generated meta-model achieved notably better performance metrics than those models based on a single encoding, and -in most cases- than those reported. The proposed method is available as a Python library for non-commercial use under the GNU General Public License (GPLv3) license.

## REQUIREMENTS
1. [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. MATLAB (R2018b used on our tests)

## HOW TO RUN
Create enviroment with conda
```shell script
conda env create -f environment.yml
```

Activate conda env
```shell script
conda activate fft_predict
```

Modify the next string values on `nextflow.config`
```shell script
dataset_path = "path/to/dataset.file"
output_dir =  "path/to/output/"
mode = "type_of_dataset" // {classification|regression}
cpus = n_cpus // number of cpus to use
```

Run the workflow
```shell script
nextflow run main.nf -resume -with-report [file name]
```
`-with-report [file name]`: creates an html file with information about execution\
`-resume`: use cached output of process to continue the execution of the pipeline

