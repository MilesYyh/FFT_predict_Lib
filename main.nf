println "----------Running FFT Peptide Predictor Pipeline----------"


/*
* Files and channels
*/

dataset_file = file(params.dataset_path)
encoding_index_ch = Channel.fromPath(params.encoding_index_dir)


process prepare_dataset {
  publishDir "${params.output_dir}/1-prepare_dataset", mode:"copy"

  input:
  path dataset_file

  output:
  path "${dataset_file.simpleName}.no_nulls.csv" into dataset_no_nulls_ch

  script:
  """
  prepare_environment.py -i $dataset_file -o ${dataset_file.simpleName}.no_nulls.csv -m ${params.mode}
  """
}


process encode_dataset_by_properties {
  tag "${encode}"
  publishDir "${params.output_dir}/2-encode_dataset/${encode}/", mode:"copy"
  

  input:
  each path(encoding_file) from encoding_index_ch
  path dataset from dataset_no_nulls_ch

  output:
  tuple val(encode), path("encoding_with_class.csv"), path("${fft_path}") into encoded_dataset_ch
  path "encoding_without_class.csv"
  path "${domain_path}"

  script:
  encode = encoding_file.simpleName
  fft_path="encoding_data_FFT.csv"
  domain_path="domain_data.csv"
  """
  encoding_dataset_using_properties.py -i $dataset -o . -e $encoding_file
  
  matlab nodisplay -nosplash -nodesktop -r \
  "addpath('${params.bin}') ;procesFourierTransform('encoding_without_class.csv', '${fft_path}', '${domain_path}'); exit;"
  """
}

process split_encoded_dataset {
  tag "${encode}"
  publishDir "${params.output_dir}/3-datasets_for_exploration/${encode}/", mode:"copy"
  
  input:
  tuple val(encode), path(dataset_class), path(dataset_fft) from encoded_dataset_ch

  output:
  path "dataset_full.csv"
  tuple val(encode), path("training_dataset.csv"), path("testing_dataset.csv") into exploration_dataset_ch, exploration_dataset_ch2

  script:
  """
  prepare_dataset_to_train_alt.py -i1 $dataset_class -i2 $dataset_fft -o . -s 1.0
  """
}


process training_dataset {
  publishDir "${params.output_dir}/4-algorithms_exploration/${encode}/", mode:"copy"
  tag "${encode}"

  input:
  tuple val(encode), path(training_dataset), path(testing_dataset) from exploration_dataset_ch 

  output:
  path "*"
  path ".command.log" 

  script:
  """
  if [[ ${params.mode} == "classification" ]]; then
    training_class_models.py -i1 $training_dataset -i2 $testing_dataset -o .

  elif [[ ${params.mode} == "regression" ]]; then
    training_regx_models.py -i1 $training_dataset -i2 $testing_dataset -o .
  fi
  """
}