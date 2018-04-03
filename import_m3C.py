import pandas as pd
import numpy as np
import timesseries_data as tm



teste = tm.ListTimeseriesM3(filename_input = 'c:\\temp\\M3Data\\data\\M3C.xls',sheet='M3Year',\
                 filename_output='c:\\temp\\M3Data\\data\\M3C.pck',forecast_period=2, frequency='Year',)


TimesSeriesYear = teste.tolist('c:\\temp\\M3Data\\data\\M3C_teste.pck')

TimesSeriesYear[0]._features_filtered_direct.columns
TimesSeriesYear[0]._fdr_level
TimesSeriesYear[1]._features_filtered_direct.columns
TimesSeriesYear[1]._fdr_level
TimesSeriesYear[2]._features_filtered_direct.columns
TimesSeriesYear[2]._fdr_level
data_sheet = pd.read_excel('c:\\temp\\M3Data\\data\\M3C.xls',sheetname='M3Year')

data_sheet.columns.values

a=data_sheet.iloc[[1]]
a=a.dropna(axis=1, how='all')
a.columns.values

series = list()
for item in a.columns:
    if isinstance(item, int):  # Check in column´s name to verify if is a integer
        series.append(float(a[item]))
final_series = np.array(series)


np.array(series).shape
valor = tm.TimeseriesM3(id_serie=a['Series'].to_string(index=False,header=False),serie= final_series ,\
                        category_name =a['Category'].to_string(index=False,header=False),\
                        frequency='Year',start_period1=a['Starting Year'].to_string(index=False,header=False),\
                        start_period2=a['Unnamed: 5'].to_string(index=False,header=False),\
                        forecast_period=2)

valor._id
valor._category_name
valor._frequency
valor._len_serie
valor._len_forecast
len(valor.get_test_data())
len(valor.get_train_data())

list((valor._serie_train))

teste = valor.get_extracted_features()
teste.columns
options = {'Year': "A"}
x = pd.Series(data=valor.get_train_data()['value'].tolist(),
              index=pd.date_range(str(valor._start_period1), periods=len(valor.get_train_data()),
                                  freq=options[valor._frequency]))

pd.date_range(str(valor._start_period1), periods=len(valor.get_train_data()),freq=options[valor._frequency])
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
df_shift, y = make_forecasting_frame(x, kind=valor._category_name, max_timeshift=valor._len_forecast, rolling_direction=1)
features_filtered_direct = extract_relevant_features(df_shift, y, column_id='id', column_value="value",fdr_level=0.15,
                                                     column_sort='time', show_warnings=False, disable_progressbar=False)

features_filtered_direct.shape[1]
b = teste.T.drop_duplicates().T
len(teste.columns)
len(b.columns)


from tsfresh import extract_features,select_features,feature_selection
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import
from tsfresh.utilities.dataframe_functions import impute


from tsfresh.utilities.dataframe_functions import make_forecasting_frame

a = valor.get_train_data()
x = pd.Series(data=valor.get_train_data()['value'].tolist(), index=pd.date_range(str(valor._start_period1), periods=len(valor.get_train_data()), freq='A'))
x.head()


df_shift, y = make_forecasting_frame(x, kind= valor._category_name, max_timeshift=6, rolling_direction=1)

X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,show_warnings=False)

X.head()
X.columns
features_filtered = feature_selection.select_features(X, y)
features_filtered.columns

from tsfresh import extract_relevant_features
features_filtered_direct = extract_relevant_features(df_shift, y,column_id='id', column_value="value",column_sort='time')
features_filtered_direct.columns
len(features_filtered_direct['value__linear_trend__attr_"slope"'])
len(x)