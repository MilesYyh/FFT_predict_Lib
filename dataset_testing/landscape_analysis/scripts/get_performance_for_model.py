import pandas as pd
import sys
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from sklearn.metrics import r2_score, accuracy_score

dataset = pd.read_csv(sys.argv[1])

response_pearson = pearsonr(dataset['Predicted'], dataset['Response'])

print(response_pearson)

#print data for image
predict_data = [value for value in dataset['Predicted']]
response_data = [value for value in dataset['Response']]

print(predict_data)
print(response_data)

print(max(predict_data))
print(min(predict_data))

print(max(response_data))
print(min(response_data))

#get values for confusion matrix
data_selected = pd.read_csv(sys.argv[2])

from sklearn.metrics import confusion_matrix
response_matrix = confusion_matrix(data_selected['Response'], data_selected['Predict'])
print(response_matrix)

print (accuracy_score(data_selected['Response'], data_selected['Predict']))