import pandas as pd
import sys
from joblib import dump, load
from sklearn import preprocessing
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.metrics import r2_score

path_data = sys.argv[1]

matrix_response = []

list_properties = ["alpha-structure_group", "betha-structure_group", "energetic_group", "hydropathy_group", "hydrophobicity_group", "index_group", "secondary_structure_properties_group", "volume_group"]

for property_value in list_properties:

	try:
		print("Using models for ", property_value)
		#read testing dataset
		testing_dataset = pd.read_csv(path_data+property_value+"/testing_dataset.csv")	
		testing_dataset = testing_dataset.drop(['response'], axis=1)

		#scale dataset
		min_max_scaler = preprocessing.MinMaxScaler()
		validation_scaler = min_max_scaler.fit_transform(testing_dataset)

		#get all list of models in meta_models
		list_models = glob.glob(path_data+property_value+"/meta_models/*.joblib")

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
response_predict_avg = []

for i in range(len(matrix_response[0])):
	point=[]
	for j in range(len(matrix_response)):
		point.append(matrix_response[j][i])
	response_predict_avg.append(np.mean(point))

#get performance compare real value v/s predicted value
pearson_value = pearsonr(response_original, response_predict_avg)[0]
spearman_value = spearmanr(response_original, response_predict_avg)[0]
kendalltau_value = kendalltau(response_original, response_predict_avg)[0]
r2_score_value = r2_score(response_original, response_predict_avg)

print("Pearson: ", pearson_value)
print("Spearman: ", spearman_value)
print("Kendall: ", kendalltau_value)
print("R2 score: ", r2_score_value)
