println "----------Running FFT Peptide Predictor Pipeline----------"


/*
* Files
*/

dataset_file = file(params.dataset_path)


process prepare_dataset {
  input:
  file dataset_file

  output:
  path "${dataset_file.simpleName}_no_nulls.csv" into dataset_no_nulls_ch

  script:
  """
  prepare_environment.py -i $dataset_file -o ${dataset_file.simpleName}_no_nulls.csv
  """
}