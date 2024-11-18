## Look-Ahead Energy Scheduling with Hierarchical Deep Learning-based Battery Degradation Model

This program implemented our proposed hierarchical deep learning-based battery degradation quantification (HDL-BDQ) model that can quantify the battery degradation given scheduled BESS daily operations. Particularly, two sequential and cohesive deep neural networks were trained to accurately estimate the degree of degradation using inputs of battery operational profiles and it can significantly outperform existing fixed or linear rate -based degradation models as well as single-stage deep neural models. Evaluations of this HDL-BDQ model were performed and validated on both a microgrid system and a bulk grid system.


### Folder/File Description
#### Folder "Daily_SCUC_Test":
* 'Case16.dat' is a sample microgrid datasheet including (Wind turbine, solar farm, BESS).
* 'SCUC_Battery_updated_BDCmethod.py' is the main optimization program for the micorgrid scheduling with the HDL-BDQ model. (The program uploaded here may not be exactly the same with the paper)
	* You can also change the trained machine learning models.
		* DNN1_IR = torch.load('DNN1_resistance.pt')
			* DNN1_CycleNum = torch.load('DNN1_cyclenumber.pt')
			* DNN2_06 = torch.load('DNN2_06.pt')

* 'DNN1_cyclenumber.pt', 'DNN1_resistance.pt', and 'DNN2_06.pt' are the trained machine learning model data HDL model), you can find more models in the trained model file.

* /environmnet:
pandas
numpy
pyomo
torch
matplotlib
itertools
time
os
gurobi solver


#### Folder "Hierarchical_Training":
* 'DNN Training.py' is designed to train a deep neural network model to predict the battery degradation value for both DNN1 and DNN2. For different models, you can just change the desired sturcture of the model (Input number, Ouput number ) and change the input datafile.
* 'DataIExtra80.xlsx', 'DataIRtoDegradation.xlsx', 'DataITtoDegradation.xlsx' and 'CapacitytoResistance.xlsx' are a sample battery data that collected processed from matlab. which is also the input for the 'readtestdata_size.py' and 'readtestdata_size_2output.py'.
* 'readtestdata_size.py' and 'readtestdata_size_2output.py' are designed to process the data from the excel and pack the data into x_sample,y_sample,x_test,y_test.
* 'x_sample.np, y_sample.np, x_test.np, y_test.np' are the tranining datas already attached. Normally it is the output of 'readtestdata_size.py'.

* Environment (Python packages):
torch
numpy
matplotlib
os

* Note the if you have a cuda supported GPU, we highly suggest you to install the cuda package to speed the training process.  https://pytorch.org/get-started/locally/
	* In our case, the training time is reduced by six times than non cuda involved traning. However, the code is also comaptiable with CPU traing (non cuda).

* Input Parameters:
	* file = 'xxxxxxxxx.xlsx'
	* sample_num = how many battery aging tests for training dataset.
	* cycle_num = how many cycles for each aging tests.
	* feature_num = number of input features
	* test_num = how many battery aging tests for testing dataset
	* seq_length = cycle_num  
	* BATCH_SIZE = trainig batch size
	* epoches = traning epoches
	* training_num = total number of traning data cell
	* validation_num = total number of validation data cell


#### Folder "Monthly_Senstivity_Test":
* 'January.dat' is a sample microgrid datasheet including (Wind turbine, solar farm, BESS). The rest of the years data are all included as well. Each month data is represented by a typical day with historical data.
* 'MonthlyTest.py' is the main optimization program for the micorgrid scheduling with the HDL-BDQ model that consider the whole years data. (The program uploaded here may not be exactly the same with the paper)
* 'MonthlyPlot.py' shows the plot of the yearly data file.
* 'DNN1_cyclenumber.pt', 'DNN1_resistance.pt', 'DNN2_06.pt' is the trained machine learning model data HDL model), you can find more models in the trained model file.

* /environmnet:
pandas
numpy
pyomo
torch
matplotlib
itertools
time
os
gurobi solver


#### Folder "Trained_DNN_Models":
This folder contains all the trained DNN models.

*If you have any questions, please reach out to __Dr. Cunzhi Zhao__ at czhao@mcneese.edu*
<br><br>

## Citation:
If you use these codes for your work, please cite the following paper:

Cunzhi Zhao and Xingpeng Li, “Cunzhi Zhao and Xingpeng Li, “Hierarchical Deep Learning Model for Degradation Prediction per Look-Ahead Scheduled Battery Usage Profile”, *IEEE Transactions on Smart Grid*, Sep. 2024.

Paper website: https://rpglab.github.io/papers/ArunR_FL-ML-R-SCUC/


## Contributions:
Cunzhi Zhao developed this program. Xingpeng Li supervised this work.


## Contact:
Cunzhi Zhao at czhao20@uh.edu.

Xingpeng Li at xli83@central.uh.edu.

Website: https://rpglab.github.io/


## License:
This work is licensed under the terms of the <a class="off" href="https://creativecommons.org/licenses/by/4.0/"  target="_blank">Creative Commons Attribution 4.0 (CC BY 4.0) license.</a>


## Disclaimer:
The author doesn’t make any warranty for the accuracy, completeness, or usefulness of any information disclosed and the author assumes no liability or responsibility for any errors or omissions for the information (data/code/results etc) disclosed.
