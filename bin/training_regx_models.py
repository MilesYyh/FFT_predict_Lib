#!/usr/bin/env python

import os
import pandas as pd
import argparse
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
from os import path
import json


# funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [
        attribute,
        statistic["mean"],
        statistic["std"],
        statistic["var"],
        statistic["max"],
        statistic["min"],
    ]

    return row


def get_data_from_dataset(dataset):

    matrix_data = []
    for i in range(len(dataset)):
        row = []
        for key in dataset.keys():
            if key != "response":
                row.append(dataset[key][i])
        matrix_data.append(row)

    return matrix_data


def main():

    args = parse_arguments()
    dataset_training = pd.read_csv(args.input_1)
    dataset_testing = pd.read_csv(args.input_2)
    path_output = args.output

    # split into dataset and class
    response_training = dataset_training["response"]
    response_testing = dataset_testing["response"]

    matrix_dataset = get_data_from_dataset(dataset_training)
    matrix_dataset_testing = get_data_from_dataset(dataset_testing)

    # generamos una lista con los valores obtenidos...
    header = ["Algorithm", "Params", "R_Score", "Pearson", "Spearman", "Kendalltau"]
    matrixResponse = []

    regx_model_save = []

    # comenzamos con las ejecuciones...

    # AdaBoost
    for loss in ["linear", "squar", "exponential"]:
        for n_estimators in [10, 100, 1000]:
            try:
                print("Excec AdaBoostRegressor with ", loss, n_estimators)
                AdaBoostObject = AdaBoost.AdaBoost(
                    matrix_dataset, response_training, n_estimators, loss
                )
                AdaBoostObject.trainingMethod()

                # get predictions for get performance
                predictions_data = AdaBoostObject.model.predict(matrix_dataset_testing)
                performanceValues = performanceData.performancePrediction(
                    response_testing, predictions_data.tolist()
                )
                pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                r_score_value = performanceValues.calculateR_score()
                params = "loss:%s-n_estimators:%d" % (loss, n_estimators)
                row = [
                    "AdaBoostClassifier",
                    params,
                    r_score_value,
                    pearsonValue,
                    spearmanValue,
                    kendalltauValue,
                ]
                matrixResponse.append(row)
                regx_model_save.append(AdaBoostObject.model)
            except:

                pass

    # Baggin
    for bootstrap in [True, False]:
        for n_estimators in [10, 100, 1000]:
            try:
                print("Excec Bagging with ", bootstrap, n_estimators)
                bagginObject = Baggin.Baggin(
                    matrix_dataset, response_training, n_estimators, bootstrap
                )
                bagginObject.trainingMethod()

                predictions_data = bagginObject.model.predict(matrix_dataset_testing)
                performanceValues = performanceData.performancePrediction(
                    response_testing, predictions_data.tolist()
                )
                pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                r_score_value = performanceValues.calculateR_score()
                params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)
                row = [
                    "Bagging",
                    params,
                    r_score_value,
                    pearsonValue,
                    spearmanValue,
                    kendalltauValue,
                ]
                print(row)
                matrixResponse.append(row)
                regx_model_save.append(bagginObject.model)
            except:
                pass

    # DecisionTree
    for criterion in ["mse", "friedman_mse", "mae"]:
        for splitter in ["best", "random"]:
            try:
                print("Excec DecisionTree with ", criterion, splitter)
                decisionTreeObject = DecisionTree.DecisionTree(
                    matrix_dataset, response_training, criterion, splitter
                )
                decisionTreeObject.trainingMethod()

                predictions_data = decisionTreeObject.model.predict(matrix_dataset_testing)

                performanceValues = performanceData.performancePrediction(
                    response_testing, predictions_data.tolist()
                )
                pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                r_score_value = performanceValues.calculateR_score()

                print(predictions_data)
                print(response_testing)
                params = "criterion:%s-splitter:%s" % (criterion, splitter)
                row = [
                    "DecisionTree",
                    params,
                    r_score_value,
                    pearsonValue,
                    spearmanValue,
                    kendalltauValue,
                ]
                print(row)
                matrixResponse.append(row)
                regx_model_save.append(decisionTreeObject.model)
            except:
                pass

    # gradiente
    for loss in ["ls", "lad", "huber", "quantile"]:
        for criterion in ["friedman_mse", "mse", "mae"]:
            for n_estimators in [10, 100, 1000]:
                try:
                    print("Excec GradientBoostingRegressor with ", loss, n_estimators, 2, 1)
                    gradientObject = Gradient.Gradient(
                        matrix_dataset,
                        response_training,
                        n_estimators,
                        loss,
                        criterion,
                        2,
                        1,
                    )
                    gradientObject.trainingMethod()

                    predictions_data = gradientObject.model.predict(matrix_dataset_testing)

                    performanceValues = performanceData.performancePrediction(
                        response_testing, predictions_data.tolist()
                    )
                    pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                    spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                    kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                    r_score_value = performanceValues.calculateR_score()

                    params = (
                        "criterion:%s-n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d"
                        % (criterion, n_estimators, loss, 2, 1)
                    )
                    row = [
                        "GradientBoostingClassifier",
                        params,
                        r_score_value,
                        pearsonValue,
                        spearmanValue,
                        kendalltauValue,
                    ]
                    print(row)
                    matrixResponse.append(row)
                    regx_model_save.append(gradientObject.model)

                except:
                    pass

    # knn
    for n_neighbors in range(2, 11):
        for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
            for metric in ["minkowski", "euclidean"]:
                for weights in ["uniform", "distance"]:
                    try:
                        print(
                            "Excec KNeighborsRegressor with ",
                            n_neighbors,
                            algorithm,
                            metric,
                            weights,
                        )
                        knnObect = knn_regression.KNN_Model(
                            matrix_dataset,
                            response_training,
                            n_neighbors,
                            algorithm,
                            metric,
                            weights,
                        )
                        knnObect.trainingMethod()

                        predictions_data = knnObect.model.predict(matrix_dataset_testing)

                        performanceValues = performanceData.performancePrediction(
                            response_testing, predictions_data.tolist()
                        )
                        pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                        spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                        kendalltauValue = performanceValues.calculatekendalltau()[
                            "kendalltau"
                        ]
                        r_score_value = performanceValues.calculateR_score()

                        params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (
                            n_neighbors,
                            algorithm,
                            metric,
                            weights,
                        )
                        row = [
                            "KNeighborsClassifier",
                            params,
                            r_score_value,
                            pearsonValue,
                            spearmanValue,
                            kendalltauValue,
                        ]
                        print(row)
                        matrixResponse.append(row)
                        regx_model_save.append(knnObect.model)
                    except:
                        pass

    # NuSVR
    for kernel in ["rbf", "linear", "poly", "sigmoid", "precomputed"]:
        for nu in [0.01, 0.05, 0.1]:
            for degree in range(3, 5):
                try:
                    print("Excec NuSVM")
                    nuSVM = NuSVR.NuSVRModel(
                        matrix_dataset, response_training, kernel, degree, 0.01, nu
                    )
                    nuSVM.trainingMethod()
                    predictions_data = nuSVM.model.predict(matrix_dataset_testing)

                    performanceValues = performanceData.performancePrediction(
                        response_testing, predictions_data.tolist()
                    )
                    pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                    spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                    kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                    r_score_value = performanceValues.calculateR_score()

                    params = "kernel:%s-nu:%f-degree:%d-gamma:%f" % (
                        kernel,
                        nu,
                        degree,
                        0.01,
                    )
                    row = [
                        "NuSVM",
                        params,
                        r_score_value,
                        pearsonValue,
                        spearmanValue,
                        kendalltauValue,
                    ]
                    matrixResponse.append(row)
                    regx_model_save.append(nuSVM.model)
                except:
                    pass


    # SVC
    for kernel in ["rbf", "linear", "poly", "sigmoid", "precomputed"]:
        for degree in range(3, 5):
            try:
                print("Excec SVM")
                svm = SVR.SVRModel(matrix_dataset, response_training, kernel, degree, 0.01)
                svm.trainingMethod()

                predictions_data = svm.model.predict(matrix_dataset_testing)

                performanceValues = performanceData.performancePrediction(
                    response_testing, predictions_data.tolist()
                )
                pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                r_score_value = performanceValues.calculateR_score()

                params = "kernel:%s-degree:%d-gamma:%f" % (kernel, degree, 0.01)
                row = [
                    "SVM",
                    params,
                    r_score_value,
                    pearsonValue,
                    spearmanValue,
                    kendalltauValue,
                ]
                matrixResponse.append(row)

                regx_model_save.append(svm.model)

            except:
                pass

    # RF
    for n_estimators in [10, 100, 1000]:
        for criterion in ["mse", "mae"]:
            for bootstrap in [True, False]:
                # try:
                print("Excec RF")
                rf = RandomForest.RandomForest(
                    matrix_dataset,
                    response_training,
                    n_estimators,
                    criterion,
                    2,
                    1,
                    bootstrap,
                )
                rf.trainingMethod()

                predictions_data = rf.model.predict(matrix_dataset_testing)
                performanceValues = performanceData.performancePrediction(
                    response_testing, predictions_data.tolist()
                )
                pearsonValue = performanceValues.calculatedPearson()["pearsonr"]
                spearmanValue = performanceValues.calculatedSpearman()["spearmanr"]
                kendalltauValue = performanceValues.calculatekendalltau()["kendalltau"]
                r_score_value = performanceValues.calculateR_score()

                params = (
                    "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s"
                    % (n_estimators, criterion, 2, 1, str(bootstrap))
                )
                row = [
                    "RandomForestRegressor",
                    params,
                    r_score_value,
                    pearsonValue,
                    spearmanValue,
                    kendalltauValue,
                ]
                matrixResponse.append(row)
                print(row)
                regx_model_save.append(rf.model)
                break
            break
        break
        # except:
        #    pass

    matrixResponseRemove = []
    for element in matrixResponse:
        if "ERROR" not in element:
            matrixResponseRemove.append(element)

    # generamos el export de la matriz convirtiendo a data frame
    dataFrameResponse = pd.DataFrame(matrixResponseRemove, columns=header)

    # generamos el nombre del archivo
    nameFileExport = path.join(path_output, "summaryProcessJob.csv")
    dataFrameResponse.to_csv(nameFileExport, index=False)

    # estimamos los estadisticos resumenes para cada columna en el header
    # instanciamos el object
    statisticObject = summaryStatistic.createStatisticSummary(nameFileExport)
    matrixSummaryStatistic = []

    # "Pearson", "Spearman", "Kendalltau"
    matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, "R_Score"))
    matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, "Pearson"))
    matrixSummaryStatistic.append(
        estimatedStatisticPerformance(statisticObject, "Spearman")
    )
    matrixSummaryStatistic.append(
        estimatedStatisticPerformance(statisticObject, "Kendalltau")
    )

    # generamos el nombre del archivo
    dataFrame = pd.DataFrame(
        matrixSummaryStatistic,
        columns=["Performance", "Mean", "STD", "Variance", "MAX", "MIN"],
    )
    nameFileExport = path.join(path_output, "statisticPerformance.csv")
    dataFrame.to_csv(nameFileExport, index=False)

    dict_summary_meta_model = {}

    print("Process extract models")
    command = f"mkdir -p {path.join(path_output, 'meta_models')}"
    print(command)
    os.system(command)

    # get max value for each performance
    for i in range(len(dataFrame)):

        max_value = dataFrame["MAX"][i]
        performance = dataFrame["Performance"][i]

        print("MAX ", max_value, "Performance: ", performance)
        information_model = {}

        information_model.update({"Performance": performance})
        information_model.update({"Value": max_value})

        # search performance in matrix data and get position
        information_matrix = []
        model_matrix = []
        algorithm_data = []

        for i in range(len(dataFrameResponse)):
            if dataFrameResponse[performance][i] == max_value:
                model_matrix.append(regx_model_save[i])
                algorithm_data.append(dataFrameResponse["Algorithm"][i])
                information_matrix.append(dataFrameResponse["Params"][i])

        array_summary = []

        for i in range(len(information_matrix)):
            model_data = {"algorithm": algorithm_data[i], "params": information_matrix[i]}
            array_summary.append(model_data)

        information_model.update({"models": array_summary})

        # export models
        for i in range(len(model_matrix)):
            name_model = path.join(path_output, f"meta_models/{performance}_model{str(i)}.joblib")
            dump(model_matrix[i], name_model)

        dict_summary_meta_model.update({performance: information_model})

    # export summary JSON file
    print("Export summary into JSON file")
    with open(path.join(path_output, "meta_models/summary_meta_models.json"), "w") as fp:
        json.dump(dict_summary_meta_model, fp)


def parse_arguments():
    """
    Parse input arguments of script

    @return: arguments parser
    """

    parser = argparse.ArgumentParser(
        "Explore multiple regression algorithms using training and testing dataset provided as inputs"
    )

    parser.add_argument(
        "-i1",
        "--input-1",
        action="store",
        required=True,
        help="training dataset file in format csv",
    )

    parser.add_argument(
        "-i2",
        "--input-2",
        action="store",
        required=True,
        help="testing dataset file in format csv",
    )

    parser.add_argument(
        "-e",
        "--encoding",
        action="store",
        help="encoding used on the input dataset",
    )

    parser.add_argument(
        "-o",
        "--output",
        action="store",
        required=True,
        help="output path",
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
