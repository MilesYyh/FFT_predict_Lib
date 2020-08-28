import sys
import os

data_input = sys.argv[1]
matlab_script = "procesFourierTransform"
list_propertyes = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

#command = "/usr/local/MATLAB/R2017a/bin/matlab -nodisplay -nosplash -nodesktop -r \"launcher_algorithm('%s', %d, %d, %d, %d, '%s'); exit;\"" % (path_output+"dataset_training.csv", p_filter, pv_filter, nmin, order, path_output+"cluster_temp.csv")
for property_data in list_propertyes:

	input_data = data_input+property_data+"/encoding_without_class.csv"
	output_data = data_input+property_data+"/encoding_data_FFT.csv"
	domain_data = data_input+property_data+"/domain_data.csv"
	command = "/usr/local/MATLAB/R2017a/bin/matlab -nodisplay -nosplash -nodesktop -r \"procesFourierTransform('%s', '%s', '%s'); exit;\"" % (input_data, output_data, domain_data)	
	print(command)
	print("--------------------------------------------")
	os.system(command)
