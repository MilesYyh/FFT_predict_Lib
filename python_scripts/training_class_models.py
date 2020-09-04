import pandas as pd
import sys
from sklearn import preprocessing
import numpy as np

from class_algorithms import AdaBoost
from class_algorithms import Baggin
from class_algorithms import BernoulliNB
from class_algorithms import DecisionTree
from class_algorithms import GaussianNB
from class_algorithms import Gradient
from class_algorithms import knn
from class_algorithms import NuSVM
from class_algorithms import RandomForest
from class_algorithms import SVM
from class_algorithms import responseTraining
from class_algorithms import summaryStatistic
from class_algorithms import performance_model
from joblib import dump, load
import json
import os

def get_data_from_dataset(dataset):

    matrix_data = []
    for i in range(len(dataset)):
        row = []
        for key in dataset.keys():
            if key != "response":
                row.append(dataset[key][i])
        matrix_data.append(row)

    return matrix_data

#funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [attribute, statistic['mean'], statistic['std'], statistic['var'], statistic['max'], statistic['min']]

    return row

dataset_training = pd.read_csv(sys.argv[1])
dataset_testing = pd.read_csv(sys.argv[2])
path_output = sys.argv[3]

#split into dataset and class
response_training = dataset_training['response']
response_testing = dataset_testing['response']

matrix_dataset = get_data_from_dataset(dataset_training)
matrix_dataset_testing = get_data_from_dataset(dataset_testing)

#explore algorithms and combinations of hyperparameters
#generamos una lista con los valores obtenidos...
header = ["Algorithm", "Params", "Validation", "Accuracy", "Recall", "Precision", "F1"]
matrixResponse = []

class_model_save = []

'''
#AdaBoost
for algorithm in ['SAMME', 'SAMME.R']:
    for n_estimators in [10,100,1000]:
        try:
            print("Excec AdaBoost with ", algorithm, n_estimators)
            AdaBoostObject = AdaBoost.AdaBoost(matrix_dataset, response_training, n_estimators, algorithm, 10)
            AdaBoostObject.trainingMethod()

            predictions = AdaBoostObject.model.predict(matrix_dataset_testing)

            metrics = performance_model.performance_model(response_testing, predictions.tolist())
            metrics.get_performance()
            params = "Algorithm:%s-n_estimators:%d" % (algorithm, n_estimators)
            row = ["AdaBoostClassifier", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
            matrixResponse.append(row)
            class_model_save.append(AdaBoostObject.model)

        except:                    
            pass

#Baggin
for bootstrap in [True, False]:
    for n_estimators in [10,100, 1000]:
        try:
            print("Excec Bagging with ", bootstrap, n_estimators)
            bagginObject = Baggin.Baggin(matrix_dataset,response_training,n_estimators, bootstrap,10)
                
            bagginObject.trainingMethod()
            params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)

            predictions = bagginObject.model.predict(matrix_dataset_testing)
            metrics = performance_model.performance_model(response_testing, predictions.tolist())
            metrics.get_performance()

            row = ["Bagging", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
            matrixResponse.append(row)
            print(row)
            class_model_save.append(bagginObject.model)
        except:
            pass


#BernoulliNB
try:
    bernoulliNB = BernoulliNB.Bernoulli(matrix_dataset, response_training, 10)
    bernoulliNB.trainingMethod()
    print("Excec Bernoulli Default Params")
    
    predictions = bernoulliNB.model.predict(matrix_dataset_testing)
    metrics = performance_model.performance_model(response_testing, predictions.tolist())
    metrics.get_performance()
    
    params = "Default"
    row = ["BernoulliNB", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
    matrixResponse.append(row)
    class_model_save.append(bernoulliNB.model)

except:
    pass

#DecisionTree
for criterion in ['gini', 'entropy']:
    for splitter in ['best', 'random']:
        try:
            print("Excec DecisionTree with ", criterion, splitter)
            decisionTreeObject = DecisionTree.DecisionTree(matrix_dataset, response_training, criterion, splitter,10)
            decisionTreeObject.trainingMethod()

            predictions = decisionTreeObject.model.predict(matrix_dataset_testing)
            metrics = performance_model.performance_model(response_testing, predictions.tolist())
            metrics.get_performance()

            params = "criterion:%s-splitter:%s" % (criterion, splitter)
            row = ["DecisionTree", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
            matrixResponse.append(row)
            class_model_save.append(decisionTreeObject.model)            
        except:
            pass
        break
    break
try:
    #GaussianNB
    gaussianObject = GaussianNB.Gaussian(matrix_dataset, response_training, 10)
    gaussianObject.trainingMethod()
    print("Excec GaussianNB Default Params")
    params = "Default"
    
    predictions = gaussianObject.model.predict(matrix_dataset_testing)
    metrics = performance_model.performance_model(response_testing, predictions.tolist())
    metrics.get_performance()
    
    row = ["GaussianNB", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
    matrixResponse.append(row)
    class_model_save.append(gaussianObject.model)    
except:
    pass

#gradiente
for loss in ['deviance', 'exponential']:
    for n_estimators in [10, 100, 1000]:
        try:
            print ("Excec GradientBoostingClassifier with ", loss, n_estimators, 2, 1)
            gradientObject = Gradient.Gradient(matrix_dataset,response_training,n_estimators, loss, 2, 1, 10)
            gradientObject.trainingMethod()
            params = "n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d" % (n_estimators, loss, 2, 1)
            
            predictions = gradientObject.model.predict(matrix_dataset_testing)
            metrics = performance_model.performance_model(response_testing, predictions.tolist())
            metrics.get_performance()
            row = ["GradientBoostingClassifier", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
            matrixResponse.append(row)
            class_model_save.append(gradientObject.model)                        
        except:
            pass

        break
    break
#knn
for n_neighbors in range(2,11):
    for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
        for metric in ['minkowski', 'euclidean']:
            for weights in ['uniform', 'distance']:
                try:
                    print("Excec KNeighborsClassifier with ", n_neighbors, algorithm, metric, weights)
                    knnObect = knn.knn(matrix_dataset, response_training, n_neighbors, algorithm, metric,  weights,10)
                    knnObect.trainingMethod()

                    predictions = knnObect.model.predict(matrix_dataset_testing)
                    metrics = performance_model.performance_model(response_testing, predictions.tolist())
                    metrics.get_performance()

                    params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (n_neighbors, algorithm, metric, weights)
                    row = ["KNeighborsClassifier", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
                    matrixResponse.append(row)
                    class_model_save.append(knnObect.model)
                
                except:
                    pass
                break
            break
        break
    break

#NuSVC
for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
    for nu in [0.01, 0.05, 0.1]:
        for degree in range(3, 15):
            try:
                print("Excec NuSVM")
                nuSVM = NuSVM.NuSVM(matrix_dataset, response_training, kernel, nu, degree, 0.01, 10)
                nuSVM.trainingMethod()

                predictions = nuSVM.model.predict(matrix_dataset_testing)
                metrics = performance_model.performance_model(response_testing, predictions.tolist())
                metrics.get_performance()

                params = "kernel:%s-nu:%f-degree:%d-gamma:%f" % (kernel, nu, degree, 0.01)
                row = ["NuSVM", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
                matrixResponse.append(row)
                class_model_save.append(nuSVM.model)

            except:
                pass
            break
        break
    break
#SVC
for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
    for C_value in [0.01, 0.05, 0.1]:
        for degree in range(3, 15):
            try:
                print ("Excec SVM")
                svm = SVM.SVM(matrix_dataset, response_training, kernel, C_value, degree, 0.01, 10)
                svm.trainingMethod()

                predictions = svm.model.predict(matrix_dataset_testing)
                metrics = performance_model.performance_model(response_testing, predictions.tolist())
                metrics.get_performance()

                params = "kernel:%s-c:%f-degree:%d-gamma:%f" % (kernel, C_value, degree, 0.01)
                row = ["SVM", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
                matrixResponse.append(row)
                class_model_save.append(svm.model)                       	

            except:
                pass
            break
        break
    break

'''
#RF
for n_estimators in [10,100,1000]:
    for criterion in ['gini', 'entropy']:
        for bootstrap in [True, False]:
            try:
                print ("Excec RF")
                rf = RandomForest.RandomForest(matrix_dataset, response_training, n_estimators, criterion, 2, 1, bootstrap, 10)
                rf.trainingMethod()

                predictions = rf.model.predict(matrix_dataset_testing)
                metrics = performance_model.performance_model(response_testing, predictions.tolist())
                metrics.get_performance()

                params = "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s" % (n_estimators, criterion, 2, 1, str(bootstrap))
                row = ["RandomForestClassifier", params, "CV-10", metrics.accuracy_value, metrics.recall_value, metrics.precision_value, metrics.f1_value]
                matrixResponse.append(row)
                class_model_save.append(rf.model)                
            except:
                pass
            break
        break
    break

#generamos el export de la matriz convirtiendo a data frame
dataFrameResponse = pd.DataFrame(matrixResponse, columns=header)

#generamos el nombre del archivo
nameFileExport =  path_output+"summaryProcess.csv"
dataFrameResponse.to_csv(nameFileExport, index=False)

#estimamos los estadisticos resumenes para cada columna en el header
#instanciamos el object
statisticObject = summaryStatistic.createStatisticSummary(nameFileExport)
matrixSummaryStatistic = []

#"Accuracy", "Recall", "Precision", "Neg_log_loss", "F1", "FBeta"
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Accuracy'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Recall'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Precision'))
matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'F1'))

#generamos el nombre del archivo
dataFrame = pd.DataFrame(matrixSummaryStatistic, columns=['Performance','Mean', 'STD', 'Variance', 'MAX', 'MIN'])
nameFileExport = path_output+"statisticPerformance.csv"
dataFrame.to_csv(nameFileExport, index=False)

print("Process extract models")
command = "mkdir -p %smeta_models" % (path_output)
print(command)
os.system(command)

dict_summary_meta_model = {}
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

    for j in range(len(dataFrameResponse)):
        if dataFrameResponse[performance][j] == max_value:
            model_matrix.append(class_model_save[j])
            algorithm_data.append(dataFrameResponse['Algorithm'][j])
            information_matrix.append(dataFrameResponse['Params'][j])

    array_summary = []

    for j in range(len(information_matrix)):
        model_data = {'algorithm': algorithm_data[j], 'params':information_matrix[j]}
        array_summary.append(model_data)

    information_model.update({"models":array_summary})

    #export models
    for j in range(len(model_matrix)):
        name_model = path_output+"meta_models/"+performance+"_model"+str(j)+".joblib"
        dump(model_matrix[j], name_model)

    dict_summary_meta_model.update({performance:information_model})

#export summary JSON file
print("Export summary into JSON file")
with open(path_output+"meta_models/summary_meta_models.json", 'w') as fp:
    json.dump(dict_summary_meta_model, fp)
