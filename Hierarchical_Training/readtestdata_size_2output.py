### 
'''
Read test data and form into a 1-D data input
'''
"""
@author: czhao29
"""



import xlrd
import numpy as np
import sys
np.set_printoptions(precision=2)
np.set_printoptions(threshold=sys.maxsize)

def read_excel(file_name1,file_name2, sample_num,training_num,feature_num, output_num,test_num, validation_num):
    wb = xlrd.open_workbook(filename=file_name1)
    wb2 = xlrd.open_workbook(filename=file_name2)
    sheet1 = wb.sheet_by_index(0)
    sheet2 = wb.sheet_by_index(1)
    sheet3 = wb.sheet_by_index(2)
    sheet4 = wb.sheet_by_index(3)
    sheet5 = wb2.sheet_by_index(0)
    sheet6 = wb2.sheet_by_index(2)
    x_sample = np.zeros((training_num,feature_num))
    y_sample = np.zeros((training_num,output_num))
    x_test = np.zeros((validation_num, feature_num))
    y_test = np.zeros((validation_num, output_num))
    Count_Sample = 0
    Count_Test = 0
    for i in range(sample_num):
        cycle_num = int(sheet1.cell(4,i+1).value)
        for j in range(cycle_num):
            column_index = i + 1       # except the title row
            row_start = 5
            row_index = row_start + j
            x_sample[Count_Sample+j, 0] = sheet5.cell(row_index,column_index).value
            x_sample[Count_Sample+j, 1] = sheet1.cell(0, column_index).value
            x_sample[Count_Sample+j, 2] = sheet1.cell(1, column_index).value
            x_sample[Count_Sample+j, 3] = sheet1.cell(2, column_index).value
            x_sample[Count_Sample+j, 4] = sheet1.cell(3, column_index).value
            #x_sample[Count_Sample+j, 5] = sheet5.cell(row_index,column_index).value #capacity
            #x_sample[Count_Sample+j, 5] = sheet1.cell(4, column_index).value
        for j in range(cycle_num):
            column_index = i + 1       # except the title row
            row_start = 5
            row_index = row_start + j            
            y_sample[Count_Sample+j,0] = sheet1.cell(row_index,column_index).value
            y_sample[Count_Sample+j,1] = sheet1.cell(4, column_index).value
        Count_Sample = Count_Sample + cycle_num    
    for i in range(test_num):
        cycle_num = int(sheet3.cell(4,i+1).value)
        for j in range(cycle_num):
            column_index = i + 1       # except the title row
            row_start = 5
            row_index = row_start + j
            x_test[Count_Test+j, 0] = sheet6.cell(row_index,column_index).value
            x_test[Count_Test+j, 1] = sheet3.cell(0, column_index).value
            x_test[Count_Test+j, 2] = sheet3.cell(1, column_index).value
            x_test[Count_Test+j, 3] = sheet3.cell(2, column_index).value
            x_test[Count_Test+j, 4] = sheet3.cell(3, column_index).value
            #x_test[Count_Test+j, 5] = sheet6.cell(row_index,column_index).value
            #x_test[Count_Test+j, 5] = sheet3.cell(4, column_index).value

        for j in range(cycle_num):
            column_index = i + 1       # except the title row
            row_start = 5
            row_index = row_start + j             
            y_test[Count_Test+j,0] = sheet3.cell(row_index,column_index).value
            y_test[Count_Test+j,1] = sheet3.cell(4, column_index).value
        Count_Test = Count_Test + cycle_num           
    return x_sample,y_sample,x_test,y_test

def filter_regularize(file_name, x_sample,y_sample,x_test,sample_num,test_num, y_test):
    wb = xlrd.open_workbook(filename=file_name)
    sheet1 = wb.sheet_by_index(0)
    sheet3 = wb.sheet_by_index(2)
    #Cycle_Max = np.amax(x_sample[:, 5])
    #Resistance_Max = np.amax(y_sample[:])
    #Resistance_Min = np.amin(y_sample[:])
    Resistance_Max = np.amax(y_sample[:, 0])
    Resistance_Min = np.amin(y_sample[:, 0])
    Cycle_Max = np.amax(y_sample[:, 1])
    #Cycle_Min = np.amin(y_sample[:, 1])
    #print(Temp_min, DISC_min, SOCL_min, SOCH_min, Type_min, Capacity_min)
    #print(Temp_max, DISC_max, SOCL_max, SOCH_max, Type_max, Capacity_max)
    Count_Sample = 0
    Count_Test = 0
    for i in range(sample_num):
        cycle_num = int(sheet1.cell(4,i+1).value)
        for j in range(cycle_num):
            #x_sample[i, j, 0] = x_sample[i, j, 0]/ (cycle_num-1)
            #x_sample[Count_Sample+j, 0] = (x_sample[Count_Sample+j, 0]-Resistance_Min)/(Resistance_Max-Resistance_Min)
            #x_sample[Count_Sample+j, 0] = x_sample[Count_Sample+j, 0]/Resistance_Max
            x_sample[Count_Sample+j, 1] = x_sample[Count_Sample+j, 1]/32
            x_sample[Count_Sample+j, 2] = x_sample[Count_Sample+j, 2]/100
            x_sample[Count_Sample+j, 3] = x_sample[Count_Sample+j, 3]/100
            x_sample[Count_Sample+j, 4] = x_sample[Count_Sample+j, 4]/200
            #x_sample[Count_Sample+j, 5] = x_sample[Count_Sample+j, 5]/Cycle_Max
            
            y_sample[Count_Sample+j,0] = (y_sample[Count_Sample+j,0]-Resistance_Min)/(Resistance_Max-Resistance_Min)
            y_sample[Count_Sample+j,1] = y_sample[Count_Sample+j,1]/Cycle_Max
            #y_sample[i,j] = y_sample[i,j]/Capacity_max
        Count_Sample = Count_Sample + cycle_num     
    for i in range(test_num):
        cycle_num = int(sheet3.cell(4,i+1).value)
        for j in range(cycle_num):
            #x_sample[i, j, 0] = x_sample[i, j, 0]/ (cycle_num-1)
            #x_test[Count_Test+j, 0] = (x_test[Count_Test+j, 0]-Resistance_Min)/(Resistance_Max-Resistance_Min)
            #x_test[Count_Test+j, 0] = x_test[Count_Test+j, 0]/Resistance_Max
            x_test[Count_Test+j, 1] = x_test[Count_Test+j, 1]/32
            x_test[Count_Test+j, 2] = x_test[Count_Test+j, 2]/100
            x_test[Count_Test+j, 3] = x_test[Count_Test+j, 3]/100
            x_test[Count_Test+j, 4] = x_test[Count_Test+j, 4]/200
            #x_test[Count_Test+j, 5] = x_test[Count_Test+j, 5]/Cycle_Max
            #y_sample[i,j] = y_sample[i,j]/Capacity_max
            y_test[Count_Test+j,0] = (y_test[Count_Test+j,0]-Resistance_Min)/(Resistance_Max-Resistance_Min)
            y_test[Count_Test+j,1] = y_test[Count_Test+j,1]/Cycle_Max
        Count_Test = Count_Test + cycle_num   
    return x_sample,y_sample,x_test,y_test





def data_shuffle(x_sample,y_sample,sample_num,cycle_num,feature_num):
    indices = np.arange(sample_num)
    np.random.shuffle(indices)
    x_sample_rd = np.zeros((sample_num, cycle_num, feature_num))
    y_sample_rd = np.zeros((sample_num, cycle_num))
    sample_index = 0
    for i in indices:
        x_sample_rd[sample_index,:,:] = x_sample[i,:,:]
        y_sample_rd[sample_index, :] = y_sample[i, :]
        sample_index += 1
    return x_sample_rd,y_sample_rd



# sample_num = 28
# feature_num = 7
# test_num = 6
# training_num = 159671 #2949728 #1009578
# validation_num = 44537# 728644 #249049

# sample_num = 28
# feature_num = 5
# test_num = 7
# output_num = 2
# training_num = 159671 #2949728 #1009578
# validation_num = 44537# 728644 #249049

# file1 = 'DataIRtoDegradation.xlsx'
# file2 = 'CapacitytoResistance.xlsx'
# x_sample,y_sample,x_test,y_test = read_excel(file1, file2,sample_num,training_num,feature_num, output_num, test_num, validation_num)
# x_sample,y_sample,x_test,y_test = filter_regularize(file1, x_sample,y_sample,x_test,sample_num,test_num, y_test)
#print(y_sample)
#x_sample,y_sample = data_shuffle(x_sample,y_sample,sample_num,cycle_num,feature_num)
