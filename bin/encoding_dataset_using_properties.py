#!/usr/bin/env python

import pandas as pd
import sys

#function to encoding 
def encoding_pca_data(sequence, data_property):

	residues = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
	sequence_encoding = []

	for element in sequence:
		#get the pos of residue
		pos = -1
		for i in range(len(residues)):
			if element == residues[i]:
				pos = i
				break
		if pos != -1:
			sequence_encoding.append(data_property['component_1'][pos])

	return sequence_encoding

dataset = pd.read_csv(sys.argv[1])
path_dir = sys.argv[2]

#check zero-padding conformation
two_base_points = []
for i in range(15):
	two_base_points.append(pow(2,i))

list_propertyes = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

for property_value in list_propertyes:
	
	print("Process property: ", property_value)
	property_dataset = pd.read_csv("../encoding_AAIndex/"+property_value+"/data_component.csv")
	dataset_encoding = []
	length_sequence = []

	for i in range(len(dataset)):

		row_data = []
		#get sequence encoding with PCA Analysis
		sequence_encoding = encoding_pca_data(dataset['sequence'][i], property_dataset)
		row_data.append(sequence_encoding)
		row_data.append(dataset['response'][i])

		dataset_encoding.append(row_data)
		length_sequence.append(len(row_data[0]))

	#make zero padding
	max_length = max(length_sequence)

	#get value near from two_base_points
	pos_pow = 0
	for i in range(len(two_base_points)):
		dif_data = two_base_points[i] - max_length
		if dif_data>=0:
			pos_pow=i
			break
	for i in range(len(dataset_encoding)):

		for j in range(len(dataset_encoding[i][0]), two_base_points[pos_pow]):
			dataset_encoding[i][0].append(0)

	#export dataset to csv
	matrix_export = []
	matrix_export_not_class = []

	for element in dataset_encoding:
		row_full = []
		row_normal = []
		for point in element[0]:
			row_full.append(point)
			row_normal.append(point)
		matrix_export_not_class.append(row_normal)
		row_full.append(element[-1])
		matrix_export.append(row_full)

	df_export = pd.DataFrame(matrix_export)
	df_export_not_class = pd.DataFrame(matrix_export_not_class)
	df_export.to_csv(path_dir+property_value+"/encoding_with_class.csv", index=False, header=False)
	df_export_not_class.to_csv(path_dir+property_value+"/encoding_without_class.csv", index=False, header=False)
