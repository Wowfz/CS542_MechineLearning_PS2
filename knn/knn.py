#!/usr/bin/env python

import csv
import random
import math
import operator
import pandas

#Code to update the data training and testing
from sys import argv

#Getting the arguments directly from the command to run the script
script, k_input, TrainData_Input, TestData_Input = argv

TrainingData = pandas.read_csv(TrainData_Input, header=None)              #creating a pandas dataframe after reading the data in the file
TestingData = pandas.read_csv(TestData_Input, header=None)

train_row_num, train_col_num = TrainingData.shape
test_row_num, test_col_num = TestingData.shape

all_predict = []

def get_euclidean_distance(dataset, row):
    total_distance = 0.00
    all_distance = []
    for i in range(train_row_num):                         #iterating through each point of the training data set
        for j in range(train_col_num - 1):
            if(isinstance(dataset.iloc[i][j], str) == True):     #checking if the feature is a real value or if it is string
                if(dataset.iloc[i][j] != row[j]):                   #each the feature of the testing dataset corresponding to the feature of the training dataset is different add 1 to the distance else add 0
                    total_distance += 1.0
                else:
                    total_distance += 0.0
            else:
                this_distance = math.pow((dataset.iloc[i][j] - row[j]), 2)       #Computation that happens when the feature is real value DL2(a,b) = sqrt(sum(ai , bi)**2))
                total_distance = total_distance + this_distance
        all_distance.append(math.sqrt(total_distance))
        total_distance = 0
    return all_distance  #returns a list of all the distance on the point of the testing dataset to each point of the training dataset

def predict(k_nearest_distance_points, TrainingData):                                    #function to find the actual predicted label associated to each testing data
    for i in range(len(k_nearest_distance_points)):                                     #go through each point of the data set corresponding the indexes of the distances
        this_predict = TrainingData.iloc[k_nearest_distance_points[i]][train_col_num - 1]     #finding all the labels
        all_predict.append(this_predict)
    return max(set(all_predict), key=all_predict.count)                          #returning the label that appears the most

def accuracy(TestingData):
    accuracy_count = 0
    i = 0
    for index, row in TestingData.iterrows():
        if(TestingData.loc[i,test_col_num-1] == TestingData.loc[i,test_col_num]):
            accuracy_count+=1
        i+=1
    print("accuracy rate = ", accuracy_count, " / ", i, " =" ,accuracy_count/i)


def main():
    TestingData[test_col_num] = TestingData[test_col_num - 1]

    i = 0
    each_points_distance = []
    k_nearest_distance_for_point = []

    for index, row in TestingData.iterrows():  # iterating through each individual row of the testing data set
        each_points_distance = get_euclidean_distance(TrainingData, row)  # finding all the distance from each of the point of the testing dataset to all the individual points  of the training datasets
        each_points_distance_sort = sorted(range(len(each_points_distance)), key=lambda k: each_points_distance[k])  # sorting the distance and getting the indexes associated to each point of the training dataset
        for i in range(int(k_input)):  # getting the number of points corresponding to the k parameter value
            k_nearest_distance_for_point.append(each_points_distance_sort[i])

        predict_result = predict(k_nearest_distance_for_point,TrainingData)  # calling the label() function to get the most recurrent label between the points selected in the training dataset
        TestingData.loc[i, test_col_num] = predict_result  # appending the predicted labels to the testing dataset

        i += 1

    TestData_knn_output =  TestData_Input + ".knnoutput"
    TestingData.to_csv(TestData_knn_output, header=False, index=False)  # writing the pandas dataframe to a file
    accuracy(TestingData)

main()