## README

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
python3 encoding_dataset_using_properties.py ../testing_data/enantioselectivity dataset_remove_nulls.csv ../testing_data/enantioselectivity/

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

## 5. SELECTION MODELS USING STATISTICAL APPROACH

## 6. CREATING META-MODEL USING SELECTED MODELS

## 7. EVALUATE PERFORMANCE META-MODEL USING VALIDATION DATASET

## RECOMENDATIONS

## CASES OF STUDY

## ABOUT US
