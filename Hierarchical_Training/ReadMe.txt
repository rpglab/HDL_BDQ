##Readme File
'DNN Training.py' is designed to train a deep neural network model to predict the battery degradation value for both DNN1 and DNN2. 
For different models, you can just change the desired sturcture of the model (Input number, Ouput number ) and change the input datafile.

'DataIExtra80.xlsx','DataIRtoDegradation.xlsx', 'DataITtoDegradation.xlsx' and 'CapacitytoResistance.xlsx' are a sample battery data that collected processed from matlab. which is also the input for the 'readtestdata_size.py' and 'readtestdata_size_2output.py'

'readtestdata_size.py' and 'readtestdata_size_2output.py' are designed to process the data from the excel and pack the data into x_sample,y_sample,x_test,y_test.

'x_sample.np ,y_sample.np ,x_test.np ,y_test.np' are the tranining datas already attached. Normally it is the output of 'readtestdata_size.py'


## Environment (Python packages)
torch
numpy
matplotlib
os

Note the if you have a cuda supported GPU, we highly suggest you to install the cuda package to speed the training process.  https://pytorch.org/get-started/locally/
	In our case, the training time is reduced by six times than non cuda involved traning. However, the code is also comaptiable with CPU traing (non cuda)


Input Parameters:
file = 'xxxxxxxxx.xlsx'
sample_num = how many battery aging tests for training dataset.
cycle_num = how many cycles for each aging tests.
feature_num = number of input features
test_num = how many battery aging tests for testing dataset
seq_length = cycle_num  
BATCH_SIZE = trainig batch size
epoches = traning epoches
training_num = total number of traning data cell
validation_num = total number of validation data cell



If you have any doubts please reach out to me at czhao@mcneese.edu
