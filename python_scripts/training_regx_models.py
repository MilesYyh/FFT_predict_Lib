import sys
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

from regx_algorithms import AdaBoost
from regx_algorithms import Baggin
from regx_algorithms import DecisionTree
from regx_algorithms import Gradient
from regx_algorithms import knn_regression
from regx_algorithms import NuSVR
from regx_algorithms import RandomForest
from regx_algorithms import SVR
from regx_algorithms import performanceData
from class_algorithms import summaryStatistic
from joblib import dump, load
import json

#funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [attribute, statistic['mean'], statistic['std'], statistic['var'], statistic['max'], statistic['min']]

    return row

dataset_training = pd.read_csv(sys.argv[1])
path_output = sys.argv[2]

#split into dataset and class
response_data = dataset_training['response']

matrix_dataset = []

for i in range(len(dataset_training)):
    row = []
    for key in dataset_training.keys():
        if key != "response":
            row.append(dataset_training[key][i])
    matrix_dataset.append(row)

dataset_scaler = pd.DataFrame(matrix_dataset)


#generamos una lista con los valores obtenidos...
header = ["Algorithm", "Params", "R_Score", "Pearson", "Spearman", "Kendalltau"]
matrixResponse = []

regx_model_save = []

#comenzamos con las ejecuciones...

#AdaBoost
for loss in ['linear', 'squar', 'exponential']:
    for n_estimators in [10, 100, 1000]:
        try:
            print("Excec AdaBoostRegressor with ", loss, n_estimators)
            AdaBoostObject = AdaBoost.AdaBoost(dataset_scaler, response_data, n_estimators, loss)
            AdaBoostObject.trainingMethod()

            #obtenemos el restante de performance
            performanceValues = performanceData.performancePrediction(response_data, AdaBoostObject.predicctions.tolist())
            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

            params = "loss:%s-n_estimators:%d" % (loss, n_estimators)
            row = ["AdaBoostClassifier", params, AdaBoostObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
            matrixResponse.append(row)
            regx_model_save.append(AdaBoostObject.model)
                   
        except:
            
            pass

#Baggin
for bootstrap in [True, False]:
    for n_estimators in [10, 100,1000]:
        try:
            print ("Excec Bagging with ",bootstrap, n_estimators)
            bagginObject = Baggin.Baggin(dataset_scaler,response_data,n_estimators, bootstrap)
            bagginObject.trainingMethod()

            performanceValues = performanceData.performancePrediction(response_data, bagginObject.predicctions.tolist())
            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

            params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)
            row = ["Bagging", params, bagginObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
            matrixResponse.append(row)
            regx_model_save.append(bagginObject.model)
            
        except:
            pass

#DecisionTree
for criterion in ['mse', 'friedman_mse', 'mae']:
    for splitter in ['best', 'random']:
        try:
            print ("Excec DecisionTree with ",criterion, splitter)
            decisionTreeObject = DecisionTree.DecisionTree(dataset_scaler, response_data, criterion, splitter)
            decisionTreeObject.trainingMethod()

            performanceValues = performanceData.performancePrediction(response_data, decisionTreeObject.predicctions.tolist())
            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

            params = "criterion:%s-splitter:%s" % (criterion, splitter)
            row = ["DecisionTree", params, decisionTreeObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
            matrixResponse.append(row)
            regx_model_save.append(decisionTreeObject.model)
            break
        except:
            pass

#gradiente
for loss in ['ls', 'lad', 'huber', 'quantile']:
    for criterion in ['friedman_mse', 'mse', 'mae']:
        for n_estimators in [10, 100,1000]:
            try:
                print ("Excec GradientBoostingRegressor with ", loss, n_estimators, 2, 1)
                gradientObject = Gradient.Gradient(dataset_scaler,response_data,n_estimators, loss, criterion, 2, 1)
                gradientObject.trainingMethod()

                performanceValues = performanceData.performancePrediction(response_data, gradientObject.predicctions.tolist())
                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                params = "criterion:%s-n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d" % (criterion, n_estimators, loss, 2, 1)
                row = ["GradientBoostingClassifier", params, gradientObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
                matrixResponse.append(row)                
                regx_model_save.append(gradientObject.model)
                
            except:
                pass


#knn
for n_neighbors in range(2,11):
    for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
        for metric in ['minkowski', 'euclidean']:
            for weights in ['uniform', 'distance']:
                try:
                    print ("Excec KNeighborsRegressor with ", n_neighbors, algorithm, metric, weights)
                    knnObect = knn_regression.KNN_Model(dataset_scaler, response_data, n_neighbors, algorithm, metric,  weights)
                    knnObect.trainingMethod()

                    performanceValues = performanceData.performancePrediction(response_data, knnObect.predicctions.tolist())
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (n_neighbors, algorithm, metric, weights)
                    row = ["KNeighborsClassifier", params, knnObect.r_score, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    regx_model_save.append(knnObect.model)
                    
                except:
                    pass


#NuSVR
for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
    for nu in [0.01, 0.05, 0.1]:
        for degree in range(3, 5):
            try:
                print ("Excec NuSVM")
                nuSVM = NuSVR.NuSVRModel(dataset_scaler, response_data, kernel, degree, 0.01, nu)
                nuSVM.trainingMethod()

                performanceValues = performanceData.performancePrediction(response_data, nuSVM.predicctions.tolist())
                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                params = "kernel:%s-nu:%f-degree:%d-gamma:%f" % (kernel, nu, degree, 0.01)
                row = ["NuSVM", params, nuSVM.r_score, pearsonValue, spearmanValue, kendalltauValue]
                matrixResponse.append(row)
                regx_model_save.append(nuSVM.model)
                
            except:
                pass

#SVC
for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
    for degree in range(3, 5):
        try:
            print ("Excec SVM")
            svm = SVR.SVRModel(dataset_scaler, response_data, kernel, degree, 0.01)
            svm.trainingMethod()

            performanceValues = performanceData.performancePrediction(response_data, svm.predicctions.tolist())
            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

            params = "kernel:%s-degree:%d-gamma:%f" % (kernel, degree, 0.01)
            row = ["SVM", params, svm.r_score, pearsonValue, spearmanValue, kendalltauValue]
            matrixResponse.append(row)
            regx_model_save.append(svm.model)
            
        except:
            pass


#RF
for n_estimators in [10,100, 1000]:
    for criterion in ['mse', 'mae']:
        for bootstrap in [True, False]:
            try:
                print ("Excec RF")
                rf = RandomForest.RandomForest(dataset_scaler, response_data, n_estimators, criterion, 2, 1, bootstrap)
                rf.trainingMethod()

                performanceValues = performanceData.performancePrediction(response_data, rf.predicctions.tolist())
                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                params = "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s" % (n_estimators, criterion, 2, 1, str(bootstrap))
                row = ["RandomForestRegressor", params, rf.r_score, pearsonValue, spearmanValue, kendalltauValue]
                matrixResponse.append(row)
                regx_model_save.append(rf.model)                
            except:
                pass

matrixResponseRemove = []
for element in matrixResponse:
    if "ERROR" not in element:
        matrixResponseRemove.append(element)

#generamos el export de la matriz convirtiendo a data frame
dataFrameResponse = pd.DataFrame(matrixResponseRemove, columns=header)

#generamos el nombre del archivo
nameFileExport = path_output+ "summaryProcessJob.csv"
dataFrameResponse.to_csv(nameFileExport, index=False)

#estimamos los estadisticos resumenes para cada columna en el header
#instanciamos el object
statisticObject = summaryStatistic.createStatisticSummary(nameFileExport)
matrixSummaryStatistic = []

#"Pearson", "Spearman", "Kendalltau"
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'R_Score'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Pearson'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Spearman'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Kendalltau'))

#generamos el nombre del archivo
dataFrame = pd.DataFrame(matrixSummaryStatistic, columns=['Performance','Mean', 'STD', 'Variance', 'MAX', 'MIN'])
nameFileExport = path_output+ "statisticPerformance.csv"
dataFrame.to_csv(nameFileExport, index=False)

dict_summary_meta_model = {}

print("Process extract models")
command = "mkdir -p %smeta_models" % (path_output)
print(command)
os.system(command)

#get max value for each performance
for i in range(len(dataFrame)):
    
    max_value = dataFrame['MAX'][i]
    performance = dataFrame['Performance'][i]

    print("MAX ", max_value, "Performance: ", performance)
    information_model = {}

    information_model.update({"Performance": performance})
    information_model.update({"Value":max_value})

    #search performance in matrix data and get position
    information_matrix = []
    model_matrix = []
    algorithm_data = []

    for i in range(len(dataFrameResponse)):
        if dataFrameResponse[performance][i] == max_value:
            model_matrix.append(regx_model_save[i])
            algorithm_data.append(dataFrameResponse['Algorithm'][i])
            information_matrix.append(dataFrameResponse['Params'][i])

    array_summary = []

    for i in range(len(information_matrix)):
        model_data = {'algorithm': algorithm_data[i], 'params':information_matrix[i]}
        array_summary.append(model_data)

    information_model.update({"models":array_summary})

    #export models
    for i in range(len(model_matrix)):
        name_model = path_output+"meta_models/"+performance+"_model"+str(i)+".joblib"
        dump(model_matrix[i], name_model)

    dict_summary_meta_model.update({performance:information_model})

#export summary JSON file
print("Export summary into JSON file")
with open(path_output+"meta_models/summary_meta_models.json", 'w') as fp:
    json.dump(dict_summary_meta_model, fp)