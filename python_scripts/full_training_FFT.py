import sys
import os

path_file = sys.argv[1]
type_response = int(sys.argv[2])

list_properties = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

print("Process path: ", path_file)

for property_data in list_properties:

	#dataset = "%s%s/training_dataset.csv" % (path_file, property_data)
	dataset_training = "%s%s/training_dataset.csv" % (path_file, property_data)
	dataset_testing = "%s%s/testing_dataset.csv" % (path_file, property_data)

	#create dir path
	command = "mkdir -p %s%s/meta_models/"  % (path_file, property_data)
	os.system(command)

	path_response = "%s%s/meta_models/" % (path_file, property_data)

	command=""
	if type_response == 1:#class
		command = "python training_class_models.py %s %s %s" % (dataset_training, dataset_testing, path_response)
	else:
		command = "python training_regx_models.py %s %s %s" % (dataset_training, dataset_testing, path_response)

	print(command)
	os.system(command)
