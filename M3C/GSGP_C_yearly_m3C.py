from M3C import timesseries_data
import pickle
import numpy as np
import pandas as pd
import pyximport; pyximport.install()
import sys
import csv
import os
from numba import jit
import quantecon as qe
from arch.bootstrap import MovingBlockBootstrap


#teste = timesseries_data.ListTimeseriesM3(filename_input = 'c:\\temp\\M3Data\\data\\M3C.xls',sheet='M3Year',\
#                 filename_output='c:\\temp\\M3Data\\data\\M3C.pck',forecast_period=6, frequency='Year',)


#timesseries_data.ListTimeseriesM3.save(listObject=times_series_year,filename_output='c:\\temp\\M3Data\\data\\M3C_teste1.pck')


def process_times_series(time_series):
    print("---------------Serie : " + str(time_series.get_id()))
    train_data=time_series.get_train_data(response=False)


    #Year Data with 6 to forecast
    length_moving = 7
    number_rows = (train_data.shape[0] // length_moving)
    names_dfs = list(map(lambda x: 'X' +x ,list(map(str,range(length_moving-1)))))
    names_dfs.append("Y")
    dfs = pd.DataFrame(columns=names_dfs)
    bs = MovingBlockBootstrap(length_moving, train_data['value'],)
    for data in bs.bootstrap(100):
        data_list = data[0][0][0:(number_rows*length_moving)]
        temp = pd.DataFrame(np.reshape(data_list, (number_rows, length_moving)),columns=names_dfs)
        for i in range(temp.shape[0]):
            dfs.loc[dfs.shape[0]+i+1] = temp.loc[i]
    dfs=dfs.drop_duplicates()
    dfs= pd.merge(dfs, train_data, left_on='Y', right_on='value')
    dfs=dfs.drop_duplicates()
    dfs = dfs.sort_values(by=['time'])
    dfs = dfs.drop(['id', 'time','value'], axis=1)

    file_path = 'C:\\temp\\M3Data\\exp\\GSGP_C\\data\\'
    #######generating file to execute GSSP_c++
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dfs,test_size=0.2, random_state=0,shuffle=False)
    #generate train file
    filename_train = file_path + 'files\\train_'+ time_series.get_id() + '.tab'
    train.to_csv(filename_train, sep="\t", quoting=csv.QUOTE_NONE,header=None,index=False)
    with open(filename_train, 'r') as original: data = original.read()
    with open(filename_train, 'w') as modified: modified.write(str(train.shape[1]-1) +"\n" + str(train.shape[0]) +"\n"+ data)
    #generate test file
    filename_test = file_path + 'files\\test_'+ time_series.get_id() + '.tab'
    test.to_csv(filename_test, sep="\t", quoting=csv.QUOTE_NONE,header=None,index=False)
    with open(filename_test, 'r') as original: data = original.read()
    with open(filename_test, 'w') as modified: modified.write(str(test.shape[1]-1) +"\n" + str(test.shape[0]) +"\n"+ data)



    ###Take the best model
    bestsMape = 200.
    filename_pred = file_path + 'files\\pred_' + time_series.get_id() + '.tab'

    names_dfs_prev = list(map(lambda x: 'X' + x, list(map(str, range(length_moving - 1)))))
    dfs_prev = test[names_dfs_prev]
    dfs_prev.to_csv(filename_pred, sep="\t", quoting=csv.QUOTE_NONE, header=None, index=False)
    with open(filename_pred, 'r') as original:
        data = original.read()
    with open(filename_pred, 'w') as modified:
        modified.write(str(dfs_prev.shape[1]) + "\n" + data)
    # Execute GSGP
    os.chdir(file_path)

    ##Loop to the best model trained
    for numIter in range(50):
        print('-----Iteration: '+ str(numIter))

        ret_GSGP = -1073741819
        while not (ret_GSGP == 0):
            ret_GSGP = os.system(file_path + "GSGP.exe -train_file " + filename_train  + " -test_file " + filename_test)

        ###generate the predict values to test data set
        #Rename the configuration file to execute
        os.system('del configuration_train.ini')
        os.system('copy configuration.ini configuration_train.ini')
        os.system('del configuration.ini')
        os.system('copy configuration_test.ini configuration.ini')

        #TODO: create e method do train e test and predict
        #Execute the unseen data
        ret_GSGP = -1073741819


        # Execute the prediction command
        ret_GSGP = -1073741819
        while not (ret_GSGP == 0):
            ret_GSGP = os.system(file_path + "GSGP.exe -test_file " + filename_pred)
        # Get the predicted value
        predicted_file_name = 'evaluation_on_unseen_data.txt'
        df_predicted = pd.read_csv(predicted_file_name, sep='\t', header=None)
        # return original configuration file
        os.system('del configuration.ini')
        os.system('copy configuration_train.ini configuration.ini')


        #calculate the SMape to test or train
        df_predicted.columns = ['pred']
        predicted_values =  df_predicted['pred'][:-1].values.tolist()
        #TODO:Vectorize the following expression
        sMape = 0
        for i in range(len(predicted_values)):
            sMape += abs(test['Y'].values[i] - predicted_values[i]) / ((abs(test['Y'].values[i])+abs(predicted_values[i]))/2)
        sMape = sMape*100/len(predicted_values)

        # If the model has a better model than the last one executed than save the files in best model
        if sMape < bestsMape:
            os.system('copy individuals.txt ' + file_path + 'bestmodel\\individuals.txt' )
            os.system('copy trace.txt ' + file_path + 'bestmodel\\trace.txt')
            bestsMape = sMape
            print('New best sMape: ' + str(bestsMape))

        #Put back the files to restart the training
        os.system('del configuration.ini')
        os.system('copy configuration_train.ini configuration.ini')

    ##Put the files from the best model to be executed
    os.system('copy ' + file_path + 'bestmodel\\individuals.txt' + ' individuals.txt ' )
    os.system('copy '+ file_path + 'bestmodel\\trace.txt' +' trace.txt' )

    ##Rename the configuration file to execute
    os.system('del configuration_train.ini')
    os.system('copy configuration.ini configuration_train.ini')
    os.system('del configuration.ini')
    os.system('copy configuration_test.ini configuration.ini')

    ##Execute the unseen data
    ret_GSGP = -1073741819
    filename_valid =file_path +  'files\\valid_'+ time_series.get_id() + '.tab'

    names_dfs_prev = list(map(lambda x: 'X' +x ,list(map(str,range(length_moving-1)))))
    dfs_prev = pd.DataFrame(columns=names_dfs_prev)

    reg_prev = time_series.get_train_data(response=False)['value'][-(length_moving-1):].values

    for i in range(length_moving-1):
        print(i)
        # Execute reg_prev with the last length_moving values
        dfs_prev= pd.DataFrame(pd.DataFrame(reg_prev).transpose())
        dfs_prev.columns = names_dfs_prev
        dfs_prev.to_csv(filename_valid, sep="\t", quoting=csv.QUOTE_NONE,header=None,index=False)
        with open(filename_valid, 'r') as original: data = original.read()
        with open(filename_valid, 'w') as modified: modified.write(str(dfs_prev.shape[1]) +"\n" + data)

        #Execute the prediction command
        ret_GSGP = -1073741819
        while not (ret_GSGP == 0):
            ret_GSGP = os.system(file_path + "GSGP.exe -test_file " + filename_valid)
        # Get the predicted value
        predicted_file_name='evaluation_on_unseen_data.txt'
        df_predicted = pd.read_csv(predicted_file_name,sep='\t',header=None)
        # Adjust the predicted value to serie and predict again
        reg_prev = reg_prev[1:len(reg_prev)]
        reg_prev = np.append(reg_prev,df_predicted.loc[0])

    # Save the value predicted
    time_series.set_forecast_data(reg_prev)
    # return original configuration file
    os.system('del configuration.ini')
    os.system('copy configuration_train.ini configuration.ini')
    return time_series


times_series_year = pickle.load(open('c:\\temp\\M3Data\\data\\M3C.pck', "rb"))


for i in range(len(times_series_year)):
    times_series_year[i] = process_times_series(times_series_year[i])
    timesseries_data.ListTimeseriesM3.save(listObject=times_series_year,filename_output='c:\\temp\\M3Data\\data\\M3C_pred_wt_constant.pck')




#
# def main():
#     # print command line arguments
#     for arg in sys.argv[1:]:
#         sys.path.append('c:\\temp\\M3Data\\M3C\\')
#         times_series_year = pickle.load(open('c:\\temp\\M3Data\\data\\M3C.pck', "rb"))
#
#         for i in range(int(sys.argv[1]),int(sys.argv[2])):
#             times_series_year[i] =  process_times_series(times_series_year[i])
#
#         timesseries_data.ListTimeseriesM3.save(listObject=times_series_year,
#                                                filename_output='c:\\temp\\M3Data\\data\\M3C_pred_'+
#                                                 str(sys.argv[3])+'.pck')
# if __name__ == "__main__":
#     main()


timesseries_year = pickle.load(open('c:\\temp\\M3Data\\data\\M3C_pred_wt_constant.pck', "rb"))

sMape = np.zeros((len(timesseries_year), 6))
for k,ts in enumerate(timesseries_year):
    real= ts.get_test_data()['value'].tolist()
    pred = ts.get_forecast_data()['value'].tolist()
    for i in range(len(pred)):
        sMape[k][i] = abs(real[i] - pred[i]) / (
                ((real[i]) + (pred[i])) / 2) * 100

import seaborn as sns
from scipy.stats import describe

import matplotlib as plt
sns.set(color_codes=True)
sMape_df = pd.DataFrame(sMape)
sMape_df.columns = list(map(lambda x: 'F' +x ,list(map(str,range(1,7)))))

sMape_df_stacked = sMape_df.melt()

sMape_df_stacked.columns
g = sns.FacetGrid(sMape_df_stacked,row="variable")
g.map(sns.distplot,'value')

sns.FacetGrid(sMape, col=[0,1])
fig, axs = sns.subplots(ncols=3,nrows=3)
sns.distplot(sMape[:,0])
sns.distplot(sMape[:,1])
sns.distplot(sMape[:,2])
sns.distplot(sMape[:,3])
sns.distplot(sMape[:,4])
sns.distplot(sMape[:,5])

describe(sMape[:,0])
describe(sMape[:,1])
describe(sMape[:,2])
describe(sMape[:,3])
describe(sMape[:,4])
describe(sMape[:,5])
np.median(sMape[:,0])



ts = timesseries_year[550]
real = ts.get_test_data()['value'].tolist()
pred = ts.get_forecast_data()['value'].tolist()




