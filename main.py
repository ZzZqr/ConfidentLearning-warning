# 导入包
import csv
import pandas as pd
import numpy as np
from cleanlab.classification import CleanLearning
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import random
import data_preprocess

ratio = 0.1
numOfFeature = 20

def reverseLabel(dataSet, ratio):
    numList = random.sample(range(0, len(dataSet)), int((len(dataSet)) * ratio))
    numList.sort()
    for num in numList:
        if dataSet.loc[num, 'final_label'] == 1:
            dataSet.loc[num, 'final_label'] = 0
        else:
            dataSet.loc[num, 'final_label'] = 1
    return dataSet, numList

def reverse_back(dataSet, index):
    tmp = dataSet
    for num in index:
        if tmp.loc[num, 'final_label'] == 1:
            tmp.loc[num, 'final_label'] = 0
        else:
            tmp.loc[num, 'final_label'] = 1
    return dataSet

def evaluate(predictSet, groundTruthSet, name):
    print(name+" Total number of Noises: " + str(len(groundTruthSet)))
    print(f"Cleanlab found {len(predictSet)} label issues.")
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for num in predictSet:
        if num in groundTruthSet:
            TP += 1
        else:
            FP += 1
    for num in groundTruthSet:
        if num in predictSet:
            TN += 1
        else:
            FN += 1
    print("TP: " + str(TP))
    print("FP: " + str(FP))
    print("TN: " + str(TN))
    print("FN: " + str(FN))
    precision = TP / (TP + FP)
    precisionLine = "Precision: " + str(precision)
    print(precisionLine)

    recall = TP / (TP + FN)
    recallLine = "Recall: " + str(recall)
    print(recallLine)

    F1 = 2 * precision * recall / (precision + recall)
    F1Line = "F1: " + str(F1)
    print(F1Line)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracyLine = "Accuracy: " + str(accuracy)
    print(accuracyLine)

    with open(r'./output_data/result.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        # rowName = "ratio" + str(ratio) + "_feature" + str(numOfFeature) + "_" + name
        data1 = [name, ratio, numOfFeature, len(groundTruthSet), len(predictSet), TP, FP, TN, FN, precision, recall, F1,
                 accuracy]
        wf.writerow(data1)

if __name__ == '__main__':
    # 载入数据
    correctDataSet = data_preprocess.readAllData()
    # 特征筛选
    # 可设置参数，反转的比例
    correctDataSet = pd.read_csv(r'./input_data/data_after_preprocessing.csv', encoding="GBK", low_memory=False)
    i = 0

    for i in range(0, 9):
        ratio = 0.1 + i * 0.05
        numOfFeature = 20
        noisyDataSet, reverseIndexSet = reverseLabel(correctDataSet, ratio)
        kk = 0
        for kk in range(0, 7):
            numOfFeature = 20 + kk * 10
            selector = SelectKBest(mutual_info_classif, k=numOfFeature)
            x_data = noisyDataSet.drop('final_label', axis=1)
            y_data = noisyDataSet['final_label']
            x_data = selector.fit_transform(x_data, y_data)
            y_data = np.array(y_data.values.tolist())

            # RF + SVM + MLP
            numCrossvalFolds = 5

            # MIX
            modelMIX = VotingClassifier(
                estimators=[('svm', svm.SVC(kernel='rbf', probability=True)),
                            ('mlp', MLPClassifier(solver='adam')),
                            ('rf', RandomForestClassifier())],
                voting='soft')
            clf = CleanLearning(modelMIX)
            # clf.fit(x_data, y_data)
            print("finish training MIX")
            label_issues = clf.find_label_issues(
                X=x_data,
                labels=y_data
            )
            label_issues.to_csv("./output_data/tableMIX2-%sratio-%sfeature.csv" % (i, numOfFeature))
            label_issues = label_issues.sort_values(by='label_quality', axis=0, ascending=[True]).reset_index(
                drop=False)
            error = []
            for j in range(0, len(label_issues)):
                if label_issues.loc[j, 'is_label_issue'] == True:
                    error.append(label_issues.loc[j, "index"])
            print(error)
            evaluate(error, reverseIndexSet, "MLP+RF+SVM")
        reverse_back(correctDataSet, reverseIndexSet)
