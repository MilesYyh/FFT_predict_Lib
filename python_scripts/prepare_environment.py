import pandas as pd
import sys
import os

#read dataset and remove outliers
print("Read csv file")
dataset = pd.read_csv(sys.argv[1])
path_outout = sys.argv[2]

print("Remove outliers")
dataset_filter = dataset.dropna()
dataset_filter.to_csv(path_outout+"dataset_remove_nulls.csv", index=False)

print("Prepare paths")
#create directory with different properties to use
list_propertyes = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

for property_value in list_propertyes:
	command = "mkdir -p %s%s" % (path_outout, property_value)
	print(command)
	os.system(command)
print("OK-Process")