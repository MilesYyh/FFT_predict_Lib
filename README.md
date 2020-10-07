## README

Predicting the effect of mutations in proteins is one of the most critical challenges in protein engineering; by knowing the effect a substitution of one (or several) residues in the protein's sequence has on its overall properties, could design a variant with a desirable function. New strategies and methodologies to create predictive models are continually being developed. However, those that claim to be general often do not reach adequate performance, and those that aim to a particular task improve their predictive performance at the cost of the method's generality. Moreover, these approaches typically require a particular decision to encode the amino acidic sequence, without an explicit methodological agreement in such endeavor. To address this issues, in this work, we applied clustering, embedding, and dimensionality reduction techniques to the AAIndex database to select meaningful combinations of physicochemical properties for the encoding stage. We then used the chosen set of properties to obtain several encodings of the same sequence, to subsequently apply the Fast Fourier Transform (FFT) on them. We perform an exploratory stage of Machine-Learning models in the frequency space, using different algorithms and hyperparameters. Finally, we select the best performing predictive models in each set of properties and create a synthetic assembled model. We extensively tested the proposed methodology on different datasets and demonstrated that the generated meta-model achieved notably better performance metrics than those models based on a single encoding, and -in most cases- than those reported. The proposed method is available as a Python library for non-commercial use under the GNU General Public License (GPLv3) license.

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
