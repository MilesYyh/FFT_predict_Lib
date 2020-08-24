import pandas as pd
import sys
from joblib import dump, load
from sklearn import preprocessing
import glob
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

path_data = sys.argv[1]

matrix_response = []

list_properties = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

for property_value in list_properties:

	
	print("Using models for ", property_value)
	#read testing dataset
	testing_dataset = pd.read_csv(path_data+property_value+"/testing_dataset.csv")	
	testing_dataset = testing_dataset.drop(['response'], axis=1)

	#scale dataset
	min_max_scaler = preprocessing.MinMaxScaler()
	validation_scaler = min_max_scaler.fit_transform(testing_dataset)

	#get all list of models in meta_models
	list_models = glob.glob(path_data+property_value+"/meta_models/*.joblib")
	
	try:	
		for model in list_models:
			load_model = load(model)
			response_predict = load_model.predict(validation_scaler)
			row_response = []
			for response in response_predict:
				row_response.append(response)
			matrix_response.append(row_response)
	except:		
		pass

#get actual response
dataset_original = pd.read_csv(path_data+list_properties[0]+"/testing_dataset.csv")
response_original = dataset_original['response']

#get mean response
response_predict_voted = []

for i in range(len(matrix_response[0])):
	point=[]
	for j in range(len(matrix_response)):
		point.append(matrix_response[j][i])
	
	unique_responses = list(set(point))

	#count for each response
	counts_data = {}
	counts_array = []
	for response in unique_responses:
		cont=0
		for element in point:
			if response==element:
				cont+=1
		counts_data.update({str(response):cont})
		counts_array.append(cont)

	max_cont= max(counts_array)

	response = -1
	for key in counts_data:
		if counts_data[key] == max_cont:
			response = int(key)
			break

	response_predict_voted.append(response)

#get performance compare real value v/s predicted value
accuracy_value = accuracy_score(response_original, response_predict_voted)
f1_value = f1_score(response_original, response_predict_voted, average='weighted')
precision_value = precision_score(response_original, response_predict_voted, average='weighted')
recall_value = recall_score(response_original, response_predict_voted, average='weighted')

print("Accuracy: ", accuracy_value)
print("Recall: ", recall_value)
print("Precision: ", precision_value)
print("F1 score: ", f1_value)
