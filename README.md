## README

## REQUIREMENTS
1. [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## HOW TO RUN
Create environment with conda
```shell script
conda env create -f environment.yml
```

Activate conda env
```shell script
conda activate fft_predict
```
Create file `nextflow.config` with the next lines:
```
params {
    dataset_path = "path/to/dataset.file"
    output_dir = "path/to/output/"
    encoding_index_dir = "encoding_AAIndex/*"     
    bin = "$PWD/bin/"
    mode = type_of_dataset"
}

executor {
  name = "local"
  cpus = n_cpus
}
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

## RECOMMENDATIONS

## CASES OF STUDY

## ABOUT US
