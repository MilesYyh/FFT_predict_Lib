import sys
import pandas as pd
import random
from sklearn import preprocessing

def read_FFT_doc(name_doc):

	matrix_data = []
	
	file_doc = open(name_doc, 'r')
	
	line = file_doc.readline()

	while line:

		data = line.replace("\n", "").split(",")
		for i in range(len(data)):
			data[i] = float(data[i])
		matrix_data.append(data)
		line = file_doc.readline()

	file_doc.close()
	return matrix_data

def get_response_from_dataset(name_doc):

	response_array = []
	
	file_doc = open(name_doc, 'r')
	
	line = file_doc.readline()

	while line:

		data = line.replace("\n", "").split(",")
		
		response_array.append(data[-1])
		line = file_doc.readline()

	file_doc.close()
	return response_array

path_data = sys.argv[1]
number_examples = int(sys.argv[2])

#make randomized array index element
index_array = [x for x in range(number_examples)]
random.shuffle(index_array)

training_len = int(number_examples*0.8)
testing_len = int(number_examples*0.2)

total = training_len+testing_len
diff = number_examples - total
testing_len = testing_len+diff

list_propertyes = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

for property_value in list_propertyes:

	print("Process property: ", property_value)
	name_doc_FFT = path_data+property_value+"/encoding_data_FFT.csv"
	name_doc_response = path_data+property_value+"/encoding_with_class.csv"

	#read data
	matrix_encoding = read_FFT_doc(name_doc_FFT)
	response_data = get_response_from_dataset(name_doc_response)

	#create dataset
	header = []
	for i in range(len(matrix_encoding[0])):
		header.append("P_"+str(i+1))
	
	#scale dataset
	min_max_scaler = preprocessing.MinMaxScaler()
	dataset_scaler = min_max_scaler.fit_transform(matrix_encoding)

	dataset = pd.DataFrame(dataset_scaler, columns=header)
	dataset['response'] = response_data	

	#export dataset
	dataset.to_csv(path_data+property_value+"/dataset_full.csv", index=False)
	
	matrix_dataset = []
	#for i in range(len(dataset)):
	#	row = matrix_encoding[i]
	#	row.append(response_data[i])
	#	matrix_dataset.append(row)

	for i in range(len(dataset)):
		row = []
		for key in dataset.keys():
			row.append(dataset[key][i])
		matrix_dataset.append(row)
		
	#create two datasets: training and testing
	matrix_dataset_training = []
	matrix_dataset_testing = []

	#preparing training
	for i in range(training_len):

		row_data = matrix_dataset[index_array[i]]	
		matrix_dataset_training.append(row_data)

	#preparing testing
	for i in range(training_len, training_len+testing_len):

		row_data = matrix_dataset[index_array[i]]
		matrix_dataset_testing.append(row_data)

	header.append("response")
	#export dataset
	dataset_testing = pd.DataFrame(matrix_dataset_testing, columns=header)
	dataset_testing.to_csv(path_data+property_value+"/testing_dataset.csv", index=False)

	dataset_training = pd.DataFrame(matrix_dataset_training, columns=header)
	dataset_training.to_csv(path_data+property_value+"/training_dataset.csv", index=False)
	
