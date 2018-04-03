import pandas as pd
from itertools import chain
import pickle
import numpy as np


class TimeseriesM3():
    def __init__(self,id_serie,serie,category_name ='',frequency='',start_period1='0',start_period2='0',forecast_period=0):
        self._id = id_serie.replace("0","").replace("N","").replace(" ","")
        self._serie = serie
        self._len_serie = len (serie)
        if forecast_period > (self._len_serie):
            raise NameError("Forecasted value is greater than the times series length.")
        self._time=list(range(0, self._len_serie, 1))
        self._len_forecast = forecast_period
        self._serie_test = self._serie[len (self._serie) - self._len_forecast:len(self._serie)]
        self._time_test = self._time[len (self._serie) - self._len_forecast:len(self._serie)]
        self._serie_train = self._serie[:-self._len_forecast]
        self._time_train = self._time[:-self._len_forecast]
        self._category_name = category_name.replace("0","").replace("N","").replace(" ","")
        self._frequency = frequency.replace("0","").replace("N","").replace(" ","")
        self._start_period1 = int(start_period1.replace("0","").replace("N","").replace(" ",""))
        self._start_period2 = int(start_period2.replace("0","").replace("N","").replace(" ",""))
        self._extract_features = False
        self._features_filtered_direct = None
        self._fdr_level = 0.05

    def get_forecastdata(self):
        pass

    def get_test_data(self):
        from itertools import chain
        return pd.DataFrame({'id':[ self._id] * len(self._serie_test),
                             'time':self._time_test,
                             'value':self._serie_test})



    def get_train_data(self):
        from itertools import chain
        return pd.DataFrame ({'id': [self._id] * len (self._serie_train) ,
                          'time': self._time_train ,
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

    def get_extracted_features(self):
        if not self._extract_features:
            from tsfresh import extract_relevant_features
            from tsfresh.utilities.dataframe_functions import make_forecasting_frame
            options = {'Year':"A"}
            x = pd.Series(data=self.get_train_data()['value'].tolist(),
                          index=pd.date_range(str(self._start_period1), periods=len(self.get_train_data()), freq=options[self._frequency]))
            df_shift, y = make_forecasting_frame(x, kind=self._category_name, max_timeshift=self._len_forecast, rolling_direction=1)
            fdr_level = [0.05,0.1,0.15,0.2,0.25]
            fdr_level_found = False
            for level in fdr_level:
                features_filtered_direct = extract_relevant_features(df_shift, y, column_id='id', column_value="value",fdr_level=level,
                                                                 column_sort='time',show_warnings=False,disable_progressbar=True)
                print('Serie: ',self._id, ' fdr_level:', level)
                if features_filtered_direct.shape[1] != 0:
                    self._fdr_level = level
                    break
            self._extract_features = True
            for i in features_filtered_direct.columns:
                features_filtered_direct[i] = features_filtered_direct[i].abs()
            self._features_filtered_direct = features_filtered_direct.T.drop_duplicates().T #to drop columns with the same value
        return self._features_filtered_direct

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
                                    frequency='Year', start_period1=raw_data['Starting Year'].to_string(index=False,header=False),
                                    start_period2=raw_data['Unnamed: 5'].to_string(index=False,header=False), \
                                    forecast_period=forecast_period)
            valor.get_extracted_features()
            time_series_list.append(valor)
            print("Times series complete: ",i)
        self.save(listObject=time_series_list,filename_output=filename_output)

    def save(self, listObject,filename_output):
        import pickle
        pickle.dump(listObject, open(filename_output, "wb"))

    def tolist(self,filename_output):
        import pickle
        listObject = pickle.load( open( filename_output, "rb" ) )
        return listObject
