##Readme File
'January.dat' is a sample microgrid datasheet including (Wind turbine, solar farm, BESS). The rest of the years data are all included as well. Each month data is represented by a typical day with historical data.


'MonthlyTest.py' is the main optimization program for the micorgrid scheduling with the HDL-BDQ model that consider the whole years data.
	#The program uploaded here may not be exactly the same with the paper.

'MonthlyPlot.py' shows the plot of the yearly data file.


'DNN1_cyclenumber.pt', 'DNN1_resistance.pt', 'DNN2_06.pt' is the trained machine learning model data HDL model), you can find more models in the trained model file.


/environmnet:
pandas
numpy
pyomo
torch
matplotlib
itertools
time
os
gurobi solver


'SCUC_Battery_updated_BDCmethod.py':
	You can also change the trained machine learning models.
	'''
	    DNN1_IR = torch.load('DNN1_resistance.pt')
            DNN1_CycleNum = torch.load('DNN1_cyclenumber.pt')
    	    DNN2_06 = torch.load('DNN2_06.pt')
	'''

If you have any doubts please reach out to me at czhao@mcneese.edu
