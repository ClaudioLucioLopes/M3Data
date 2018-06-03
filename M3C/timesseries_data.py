import pandas as pd
from itertools import chain
import pickle
import numpy as np


class TimeseriesM3():
    def __init__(self,id_serie,serie,category_name ='',frequency='',start_period1='0',start_period2='0',forecast_period=0):
        self._id = id_serie.replace("N","").replace(" ","")
        self._serie = serie
        self._len_serie = len (serie)
        if forecast_period > (self._len_serie):
            raise NameError("Forecasted value is greater than the times series length.")
        self._time=list(range(0, self._len_serie, 1))
        self._len_forecast = forecast_period
        self._serie_test = self._serie[len (self._serie) - self._len_forecast:len(self._serie)]
        self._serie_forecasted = self._serie[len(self._serie) - self._len_forecast:len(self._serie)]
        self._time_test = self._time[len (self._serie) - self._len_forecast:len(self._serie)]
        self._serie_train = self._serie[:-self._len_forecast]
        self._time_train = self._time[:-self._len_forecast]
        self._category_name = category_name.replace(" ","")
        self._frequency = frequency.replace(" ","")
        self._start_period1 = int(start_period1.replace(" ",""))
        if self._start_period1 == 0:
            self._start_period1 = pd.Timestamp.min.year+1
        self._start_period2 = int(start_period2.replace(" ",""))
        self._extract_features = False
        self._features_filtered_direct = None
        self._fdr_level = 0.05
        self._foptions = {'Year': "A",'Quarter':"Q",'Month':"M"}

    def get_forecast_data(self,response=False):
        if response :
            return pd.DataFrame({'value': self._serie_forecasted}).as_matrix().flatten()
        else:
            return pd.DataFrame({'id':[ self._id] * len(self._serie_test),
                             'time':pd.date_range(str(max(self.get_train_data()['time']) + pd.offsets.DateOffset(years=1)),\
                                                  periods=len(self._serie_test),
                                                freq=self._foptions[self._frequency]),\
                             'value':self._serie_forecasted})


    def set_forecast_data(self,forecast_values):
        self._serie_forecasted = np.array(forecast_values)

    def get_test_data(self,response=False):
        from itertools import chain
        if response :
            return pd.DataFrame({'value': self._serie_test}).as_matrix().flatten()
        else:
            return pd.DataFrame({'id':[ self._id] * len(self._serie_test),
                             'time':pd.date_range(str(max(self.get_train_data()['time']) + pd.offsets.DateOffset(years=1)),\
                                                  periods=len(self._serie_test),
                                                freq=self._foptions[self._frequency]),\
                             'value':self._serie_test})



    def get_train_data(self,response=False):
        if response :
            return pd.DataFrame({'value': self._serie_train}).as_matrix().flatten()
        else:
            return pd.DataFrame ({'id': [self._id] * len (self._serie_train) ,
                          'time': pd.date_range(str(self._start_period1), periods=len(self._serie_train),
                                                freq=self._foptions[self._frequency]),
                                  'value': self._serie_train})

    def get_id(self):
        return self._id

    def alter_forescast_period(self,forecast_period=0):
        if forecast_period <= 0 :
            raise NameError ("Forecasted period should be greater than 0.")
        if forecast_period >= (self._len_serie):
            raise NameError("Forecasted period is greater than or equal to the times series length.")
        self._len_forecast = forecast_period
        self._serie_test = self._serie[len (self._serie) - self._len_forecast:len (self._serie)]
        self._serie_train = self._serie[:-self._len_forecast]
        self._time_test = self._time[len (self._serie) - self._len_forecast:len (self._serie)]
        self._time_train = self._time[:-self._len_forecast]
        self._extract_features = False

    def get_extracted_features(self,train_data):
        if not self._extract_features:
            data_serie = pd.concat([self.get_train_data(), self.get_test_data()])
            from tsfresh import extract_relevant_features
            from tsfresh.utilities.dataframe_functions import make_forecasting_frame

            x = pd.Series(data=data_serie['value'].tolist(),
                          index=pd.date_range(str(self._start_period1), periods=len(data_serie), freq=self._foptions[self._frequency]))
            df_shift, y = make_forecasting_frame(x, kind=self._category_name, max_timeshift=self._len_forecast, rolling_direction=1)
            fdr_level = [0.05,0.1,0.15,0.2,0.25]
            fdr_level_found = False
            for level in fdr_level:
                features_filtered_direct = extract_relevant_features(df_shift, y, column_id='id', column_value="value",fdr_level=level,
                                                                 column_sort='time',show_warnings=False,disable_progressbar=True)
                print('Serie: ',self._id, ' fdr_level:', level,'features_filtered_direct: ', features_filtered_direct.shape)
                if features_filtered_direct.shape[1] != 0:
                    self._fdr_level = level
                    break
            self._extract_features = True
            for i in features_filtered_direct.columns:
                features_filtered_direct[i] = features_filtered_direct[i].abs()#TODO:improve a way to extract similar columns
            self._features_filtered_direct = features_filtered_direct.T.drop_duplicates().T #to drop columns with the same value
        if train_data:
            return self._features_filtered_direct.iloc[0:self._features_filtered_direct.shape[0] - self._len_forecast]
        else:
            return self._features_filtered_direct.iloc[self._features_filtered_direct.shape[0]- self._len_forecast:self._features_filtered_direct.shape[0]]

class ListTimeseriesM3():
    def __init__(self, filename_input='',sheet='',filename_output='',forecast_period=0,frequency=''):
        self._filename_input = filename_input
        data_sheet = pd.read_excel(filename_input, sheetname=sheet)
         #Loop to iterate in the list
        time_series_list=[]
        for i in range(0,len(data_sheet)):
            raw_data = data_sheet.iloc[[i]].dropna(axis=1, how='all')
            #For each series we are going to treat
            series = list()
            for item in raw_data.columns:
                if isinstance(item, int):  # Check in column´s name to verify if is a integer
                    series.append(float(raw_data[item]))
            final_series = np.array(series)
            valor = TimeseriesM3(id_serie=raw_data['Series'].to_string(index=False,header=False), serie=final_series,
                                    category_name=raw_data['Category'].to_string(index=False,header=False), \
                                    frequency=frequency, start_period1=raw_data[raw_data.columns[4]].to_string(index=False,header=False),
                                    start_period2=raw_data[raw_data.columns[5]].to_string(index=False,header=False), \
                                    forecast_period=forecast_period)
            valor.get_extracted_features(train_data = True)
            time_series_list.append(valor)
            print("Times series complete: ",i)
            ListTimeseriesM3.save(listObject=time_series_list,filename_output=filename_output)

    def save(listObject,filename_output):
        import pickle
        pickle.dump(listObject, open(filename_output, "wb"))

