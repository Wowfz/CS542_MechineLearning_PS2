#process and update the missing data in dataset
import pandas
import numpy
from sys import argv

#arguments for the command
script, TrainData_Input, TestData_Input = argv

# process data
def process(data_input):
	data_input = data_input.replace('?', numpy.NaN)

	letter_col_miss = [0,3,4,5,6]	#letter-valued features col with missing data
	for i in letter_col_miss:
		data_input[i] = data_input[i].fillna(data_input[i].mode()[0])			#replacing letter missing data with a or b

	float_col_miss = [1,13]					#float-valued features col with missing data
	labels = ['+', '-']
	for col in float_col_miss:
		data_input[col] = data_input[col].apply(float)		#convert strings values into float
		for c in labels:
			data_input.loc[(data_input[col].isnull()) & (data_input[15] == c), col] = data_input[col][data_input[15] == c].mean()  #calculate the ? according to the equation

	float_col = [1,2,7,10,13,14] 	#all of the float-valued features col
	for col in float_col:
		data_input[col] = (data_input[col] - data_input[col].mean()) / data_input[col].std()				#process the data according to the rule

	return data_input

TrainingData = pandas.read_csv(TrainData_Input, header=None)
TestingData = pandas.read_csv(TestData_Input, header=None)
TrainingData = process(TrainingData)
TestingData = process(TestingData)
TrainingData.to_csv('crx.training.processed', header=False, index=False)
TestingData.to_csv('crx.testing.processed', header=False, index=False)
