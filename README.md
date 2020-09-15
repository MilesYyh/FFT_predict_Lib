## README

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

Modifify the next string values on `nextflow.config`
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

## WORKFLOW

## 1. PREPARE ENVIRONMENT

Prepare environment used to training predictive meta-model using FFT

Script to execute:


```
python3 prepare_environment.py dataset_input.csv path_output
```

NOTE: dataset_input.csv should be formatted with relative or absolute path

Example of use:

```
python3 prepare_environment.py ../testing_data/enantioselectivity/enantioselectivity.csv ../testing_data/enantioselectivity/

```

## 2. ENCODING PROCESS

Encoding dataset without null values using the 8 meta-properties saved into encoding_AAIndex.

Script to execute:

```
python3 encoding_dataset_using_properties.py dataset_with_remove_nulls.csv path_to_save_encodings
```

NOTE: The path to use must be the same used to save the results of prepare environment.

Example of use:

```
python3 encoding_dataset_using_properties.py ../testing_data/enantioselectivity/dataset_remove_nulls.csv ../testing_data/enantioselectivity/

```

In a second step, you must apply Fast Fourier Transformation using script process_FFT_encodings.py

Script to execute:

```
python3 process_FFT_encoding path_to_encoding_data
```

A relevant point associated to this script is the use of Matlab script to implement the process

NOTE: The path to use must be the same used to save the results of prepare environment.

Example of use:

```
python3 process_FFT_encoding.py ../testing_data/enantioselectivity/
```
## 3. PREPARE DATASET TO EXPLORATION MACHINE LEARNING METHODS

To prepare datase for training process, you must apply the script prepare_dataset_to_train.py. This script split the dataset into two datasets: Training and Validation with the 80% and 20% of size, respectively. The division is randomly and is persistent for each property using during the process.

Execution script:

```
python3 prepare_dataset_to_train.py path_to_Data_digitized number_examples
```

The dataset is the same previously used and the number of examples is a natural number.

Example of use:

```
python3 prepare_dataset_to_train.py ../testing_data/enantioselectivity/ 152
```
## 4. EXPLORING DATASET USING MACHINE LEARNING ALGORITHM

This step allows exploring and selecting the best combinations of algorithms and hyperparameters based on machine learning algorithm and time series predictors algorithms. This step using two scripts: training_class_models.py for classification models and training_regx_models.py for predictivide models. 

The general scripts is full_training_FFT.py, the execution of this algorithm is

```
python3 full_training_FFT.py path_to_datasets type_response
```

Example of use

```
python3 full_training_FFT.py ../testing_data/enantioselectivity/ 2
```

## 5. SELECTION MODELS USING STATISTICAL APPROACH

The selection of the models is based on application of statistical techniques....

## 6. CREATING META-MODEL USING SELECTED MODELS AND EVALUATE PERFORMANCE

For creating meta-models, depends of the type of response, you must use the scripts make_meta_model_XXX.py where XXX corresponding to class or regx, depend of the type of meta-model to build.

The execution of these scripts is:

```
python3 make_meta_model_class.py path_to_datasets
```

The path_data is the same used previuosly

## RECOMENDATIONS

## CASES OF STUDY

## ABOUT US
