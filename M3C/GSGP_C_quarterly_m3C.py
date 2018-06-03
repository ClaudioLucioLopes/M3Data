import pickle
import numpy as np
import pandas as pd
import pyximport; pyximport.install()
import sys
sys.path.append('/home/claudio/PycharmProjects/M3Data/M3C')
import timesseries_data as tm
import csv
import os
from arch.bootstrap import MovingBlockBootstrap

# teste = tm.ListTimeseriesM3(filename_input = '/home/claudio/PycharmProjects/M3Data/data/M3C.xls',sheet='M3Quart',\
#                  filename_output='/home/claudio/PycharmProjects/M3Data/data/M3Quart.pck',forecast_period=8, frequency='Quarter',)
#


#timesseries_data.ListTimeseriesM3.save(listObject=times_series_year,filename_output='c:\\temp\\M3Data\\data\\M3C_teste1.pck')

def process_times_series(time_series):
    print("---------------Serie : " + str(time_series.get_id()))
    train_data=time_series.get_train_data(response=False)


    #Year Data with 6 to forecast
    length_moving = 9
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

    file_path = '/home/claudio/PycharmProjects/M3Data/exp/GSGP_C/data/'
    #######generating file to execute GSSP_c++
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(dfs,test_size=0.2, random_state=0,shuffle=False)
    #generate train file
    filename_train =  'train_'+ time_series.get_id() + '.tab'
    train.to_csv(filename_train, sep="\t", quoting=csv.QUOTE_NONE,header=None,index=False)
    with open(filename_train, 'r') as original: data = original.read()
    with open(filename_train, 'w') as modified: modified.write(str(train.shape[1]-1) +"\n" + str(train.shape[0]) +"\n"+ data)
    #generate test file
    filename_test =  'test_'+ time_series.get_id() + '.tab'
    test.to_csv(filename_test, sep="\t", quoting=csv.QUOTE_NONE,header=None,index=False)
    with open(filename_test, 'r') as original: data = original.read()
    with open(filename_test, 'w') as modified: modified.write(str(test.shape[1]-1) +"\n" + str(test.shape[0]) +"\n"+ data)



    ###Take the best model
    bestsMape = 200.
    filename_pred =  'pred_' + time_series.get_id() + '.tab'

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
            ret_GSGP = os.system(file_path + "GP -train_file " + filename_train + " -test_file " + filename_test)

        ###generate the predict values to test data set
        #Rename the configuration file to execute
        os.system('rm configuration_train.ini')
        os.system('cp configuration.ini configuration_train.ini')
        os.system('rm configuration.ini')
        os.system('cp configuration_test.ini configuration.ini')

        #TODO: create e method do train e test and predict
        #Execute the unseen data
        ret_GSGP = -1073741819


        # Execute the prediction command
        ret_GSGP = -1073741819
        while not (ret_GSGP == 0):
            ret_GSGP = os.system(file_path + "GP -test_file " + filename_pred)
        # Get the predicted value
        predicted_file_name = 'evaluation_on_unseen_data.txt'
        df_predicted = pd.read_csv(predicted_file_name, sep='\t', header=None)
        # return original configuration file
        os.system('rm configuration.ini')
        os.system('cp configuration_train.ini configuration.ini')


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
            os.system('cp individuals.txt ' + file_path + 'bestmodel/individuals.txt' )
            os.system('cp trace.txt ' + file_path + 'bestmodel/trace.txt')
            bestsMape = sMape
            print('New best sMape: ' + str(bestsMape))

        #Put back the files to restart the training
        os.system('rm configuration.ini')
        os.system('cp configuration_train.ini configuration.ini')

    ##Put the files from the best model to be executed
    os.system('cp ' + file_path + 'bestmodel/individuals.txt' + ' individuals.txt' )
    os.system('cp ' + file_path + 'bestmodel/trace.txt' +' trace.txt' )

    ##Rename the configuration file to execute
    os.system('rm configuration_train.ini')
    os.system('cp configuration.ini configuration_train.ini')
    os.system('rm configuration.ini')
    os.system('cp configuration_test.ini configuration.ini')

    ##Execute the unseen data
    ret_GSGP = -1073741819
    filename_valid =file_path +  'valid_'+ time_series.get_id() + '.tab'

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
            ret_GSGP = os.system(file_path + "GP -test_file " + filename_valid)
        # Get the predicted value
        predicted_file_name='evaluation_on_unseen_data.txt'
        df_predicted = pd.read_csv(predicted_file_name,sep='\t',header=None)
        # Adjust the predicted value to serie and predict again
        reg_prev = reg_prev[1:len(reg_prev)]
        reg_prev = np.append(reg_prev,df_predicted.loc[0])

    # Save the value predicted
    time_series.set_forecast_data(reg_prev)
    # return original configuration file
    os.system('rm configuration.ini')
    os.system('cp configuration_train.ini configuration.ini')
    return time_series


times_series_M3Quart= pickle.load(open('/home/claudio/PycharmProjects/M3Data/data/M3Quart.pck', "rb"))



for i in range(len(times_series_M3Quart)):
    times_series_M3Quart[i] = process_times_series(times_series_M3Quart[i])
    tm.ListTimeseriesM3.save(listObject=times_series_M3Quart,filename_output='/home/claudio/PycharmProjects/M3Data/data/M3Quart_pred_wt_constant.pck')




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

#Year Data with 6 to forecast
length_moving = 9
timesseries_year = pickle.load(open('/home/claudio/PycharmProjects/M3Data/data/M3C_pred_quaterly_final_20180508.pck', "rb"))

sMape = np.zeros((len(timesseries_year), length_moving-1))
for k,ts in enumerate(timesseries_year):
    real= ts.get_test_data()['value'].tolist()
    pred = ts.get_forecast_data()['value'].tolist()
    for i in range(len(pred)):
        sMape[k][i] = abs(real[i] - pred[i]) / (
                ((real[i]) + (pred[i])) / 2) * 100


from scipy.stats import describe
for i in range(length_moving-1):
    print(i+1,describe(sMape[:, i]))
describe(np.concatenate((sMape[:, 0],sMape[:, 1],sMape[:, 2],sMape[:, 3])))
describe(np.concatenate((sMape[:, 0],sMape[:, 1],sMape[:, 2],sMape[:, 3],sMape[:, 4],sMape[:, 5])))
describe(np.concatenate((sMape[:, 0],sMape[:, 1],sMape[:, 2],sMape[:, 3],sMape[:, 4],sMape[:, 5],sMape[:, 6],sMape[:, 7])))


import matplotlib as plt
import seaborn as sns
sns.set_style("whitegrid")
sMape_df = pd.DataFrame(sMape)
sMape_df.columns = list(map(lambda x: ' ' +x ,list(map(str,range(1,length_moving )))))

sMape_df_stacked = sMape_df.melt()
sMape_df_stacked=sMape_df_stacked.rename(columns={'variable': 'Forecast'})
g = sns.FacetGrid(sMape_df_stacked,col="Forecast",col_wrap=2)
g.map(sns.distplot,'value')






