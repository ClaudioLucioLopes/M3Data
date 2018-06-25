import numpy as np
import pandas as pd
import pyximport; pyximport.install()
import sys
import xlrd
import pickle
from M3C import timesseries_data as tm
sys.path.append('c:\\temp\\M3Data\\M3C ')

path ='c:\\temp\\M3Data\\data\\'
file = 'M3Forecast.xls'
filename_input = path+file
xls = xlrd.open_workbook(filename_input, on_demand=True)
methods_names = xls.sheet_names()
all_forecast_methods=list()
for sheet in methods_names:
    #if sheet == 'NAIVE2':
    time_series_list = []
    data_sheet = pd.read_excel(filename_input, sheetname=sheet, header=None, index_col=None, )
    data_sheet.rename(columns={0: 'id'}, inplace=True)
    data_sheet.rename(columns={1: 'forecast_len'}, inplace=True)
    data_sheet.columns
    for i in range(0, len(data_sheet)):
        raw_data = data_sheet.iloc[[i]].dropna(axis=1, how='all')
        series = list()
        for item in raw_data.columns.values:
            if isinstance(item, np.int64):  # Check in column´s name to verify if is a integer
                series.append(float(raw_data[item]))
        final_series = np.array(series)
        valor = tm.TimeseriesM3(id_serie=raw_data.id.to_string(index=False), serie=final_series,
                                category_name='Forecast', \
                                frequency='Year', start_period1='0',
                                start_period2='0', \
                                forecast_period=-1 * int(raw_data.forecast_len.to_string(index=False)))
        time_series_list.append(valor)
    all_forecast_methods.append(time_series_list)

#Verifing the values predict. The AAM1 and AAM2 methods have just 2184 values adjusted
for forecast in all_forecast_methods:
    print(len(forecast))
len(methods_names)

time_series_list[0].get_train_data(response=False)
filename_output = path +'\\all_forecast_methods.pck'
pickle.dump(all_forecast_methods, open(filename_output, "wb"))



valor.get_train_data(response=True)
valor.get_id()



path ='c:\\temp\\M3Data\\data\\'
file = 'M3C.xls'
filename_input = path+file
ts_others = tm.ListTimeseriesM3(filename_input = filename_input,sheet='M3Other',\
                    filename_output=path+'M3C_pred_other.pck',forecast_period=8, frequency='Month',)

ts_monthly = tm.ListTimeseriesM3(filename_input = filename_input,sheet='M3Month',\
                 filename_output=path+'M3C_pred_Month.pck',forecast_period=18, frequency='Month',)

ts_quarterly = tm.ListTimeseriesM3(filename_input = filename_input,sheet='M3Quart',\
                  filename_output=path+'M3C_pred_Quart.pck',forecast_period=8, frequency='Quarter',)

ts_yearly = tm.ListTimeseriesM3(filename_input = filename_input,sheet='M3Year',\
                filename_output=path+'M3C_pred_yearly.pck',forecast_period=6, frequency='Year',)





filename_output = path +'\\all_forecast_methods.pck'
all_forecast_methods = pickle.load(open(filename_output, "rb"))


##Other Id: 2830 to 3003

##Year Id: 1 to 645
ts_real_yearly= pickle.load(open(path+'M3C_pred_yearly.pck', "rb"))
from scipy.stats import describe
year_forecast=6
yearly_forecast_methods = all_forecast_methods[0:22]
with open(path + 'yearly.csv', 'a') as file:
    for fnum,forecast_method in enumerate(yearly_forecast_methods):
        sMape = np.zeros((len(ts_yearly), year_forecast))
        for k in range(0, 645):
            real = ts_real_yearly[k].get_test_data()['value'].tolist()
            pred = forecast_method[k].get_train_data()['value'].tolist()
            for i in range(len(pred)):
                sMape[k][i] = abs(real[i] - pred[i]) / (
                        ((real[i]) + (pred[i])) / 2) * 100

        file.write(methods_names[fnum] + ',')
        for i in range(year_forecast):
            file.write(str(round(describe(sMape[:, i])[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3])))[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5])))[2],2)))
        file.write('\n')


##Quaterly Id: 646 to 1401
ts_real_quarterly = pickle.load(open(path+'M3C_pred_Quart.pck', "rb"))
from scipy.stats import describe
period_forecast=8
quarterly_forecast_methods = all_forecast_methods
with open(path + 'quarterly.csv', 'a') as file:
    for fnum,forecast_method in enumerate(quarterly_forecast_methods):
        sMape = np.zeros((len(ts_real_quarterly), period_forecast))
        for k in range(0, 756):
            real = ts_real_quarterly[k].get_test_data()['value'].tolist()
            if fnum in (22,23):
                pred = forecast_method[k].get_train_data()['value'].tolist()
            else:
                pred = forecast_method[k+645].get_train_data()['value'].tolist()
            for i in range(len(pred)):
                sMape[k][i] = abs(real[i] - pred[i]) / (
                        ((real[i]) + (pred[i])) / 2) * 100

        file.write(methods_names[fnum] + ',')
        for i in range(period_forecast):
            file.write(str(round(describe(sMape[:, i])[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3])))[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5], sMape[:, 6], sMape[:, 7])))[2],2)))
        file.write('\n')


##Monthly Id: 1402 to 2829
ts_real_monthly = pickle.load(open(path+'M3C_pred_Month.pck', "rb"))
from scipy.stats import describe
period_forecast=18
monthly_forecast_methods = all_forecast_methods
with open(path + 'monthly.csv', 'a') as file:
    for fnum,forecast_method in enumerate(monthly_forecast_methods):
        sMape = np.zeros((len(ts_real_monthly), period_forecast))
        for k in range(0, 1428):
            real = ts_real_monthly[k].get_test_data()['value'].tolist()
            if fnum in (22,23):
                pred = forecast_method[k+756].get_train_data()['value'].tolist()
            else:
                pred = forecast_method[k+1401].get_train_data()['value'].tolist()
            for i in range(len(pred)):
                sMape[k][i] = abs(real[i] - pred[i]) / (
                        ((real[i]) + (pred[i])) / 2) * 100

        file.write(methods_names[fnum] + ',')
        for i in range(period_forecast):
            file.write(str(round(describe(sMape[:, i])[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3])))[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5], sMape[:, 6], sMape[:, 7])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5],sMape[:, 6], sMape[:, 7], sMape[:, 8], sMape[:, 9], sMape[:, 10], sMape[:, 11])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5],sMape[:, 6], sMape[:, 7], sMape[:, 8], sMape[:, 9], sMape[:, 10], sMape[:, 11],sMape[:, 12], sMape[:, 13], sMape[:, 14])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5],sMape[:, 6], sMape[:, 7], sMape[:, 8], sMape[:, 9], sMape[:, 10], sMape[:, 11],sMape[:, 12], sMape[:, 13], sMape[:, 14], sMape[:, 15], sMape[:, 16], sMape[:, 17])))[2],2)))
        file.write('\n')


##Other Id: 2830 to 3003
ts_real_other = pickle.load(open(path+'M3C_pred_other.pck', "rb"))
from scipy.stats import describe
period_forecast=8
other_forecast_methods = all_forecast_methods[0:22]
with open(path + 'other.csv', 'a') as file:
    for fnum,forecast_method in enumerate(other_forecast_methods):
        sMape = np.zeros((len(ts_real_other), period_forecast))
        for k in range(0, 174):
            real = ts_real_other[k].get_test_data()['value'].tolist()
            pred = forecast_method[k+2829].get_train_data()['value'].tolist()
            for i in range(len(pred)):
                sMape[k][i] = abs(real[i] - pred[i]) / (
                        ((real[i]) + (pred[i])) / 2) * 100

        file.write(methods_names[fnum] + ',')
        for i in range(period_forecast):
            file.write(str(round(describe(sMape[:, i])[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3])))[2], 2)) + ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5])))[2],2))+ ',')
        file.write(str(round(describe(np.concatenate((sMape[:, 0], sMape[:, 1], sMape[:, 2], sMape[:, 3], sMape[:, 4], sMape[:, 5], sMape[:, 6], sMape[:, 7])))[2],2)))
        file.write('\n')

