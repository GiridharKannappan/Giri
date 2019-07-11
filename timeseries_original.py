# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:29:01 2019

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 02:21:08 2019

@author: User
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import seaborn as sns
from matplotlib import rcParams
from IPython.display import HTML,display,display_html
rcParams['figure.figsize']=11,5
data1=pd.read_excel('G:\prog\datasss.xls','Sheet3')
#data2=pd.read_csv('AirPassengers.csv')
data1=data1.drop(['Death'],axis=1)
#data2['TravelDate']=pd.to_datetime(data2['TravelDate'],infer_datetime_format=True)
data1['Date']=pd.to_datetime(data1['Date'],infer_datetime_format=True)
indexedDataset=data1.set_index(['Date'])
#indexedDataset=data2.set_index(['TravelDate'])
plt.plot(indexedDataset,color='green',marker='*',markersize=3,label='Cases')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('People Affected')
plt.title('Degen Fever Forecasting')
plt.show()
rolmean=indexedDataset.rolling(window=12).mean()
rolstd=indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)
org=plt.plot(indexedDataset,color='grey',label='original')
mean=plt.plot(rolmean,color='blue',label='mean')
std=plt.plot(rolstd,color='black',label='std')
plt.legend(loc='best')
plt.xlabel('Year From 2003-2018')
plt.ylabel('People Affected')
plt.title('Rolling Mean and Standard Deviation')
plt.show()
print("The std and mean should be constant the graph is not like that so it is not stationary")
#AGUMENTED DICKY FULLER TEST
from statsmodels.tsa.stattools import adfuller
def dickey(Dataset,ts):
    std=Dataset[ts].rolling(window=12).std()
    mean=Dataset[ts].rolling(window=12).mean()
    org=plt.plot(Dataset,color='blue',label='Orginal')
    mean=plt.plot(mean,color='red',label='Mean')
    std=plt.plot(std,color='green',label='standard Deviation')
    plt.legend(loc='best')
    plt.xlabel('Year from 2003-2018')
    plt.ylabel('People Affected')
    plt.title('Rolling Mean and STD')
    #NULL HYPOTHESIS: NOT STATIONARY
    #ALTERNATE HYPOTHESIS:STATIONARY
    DF_TEST=adfuller(Dataset[ts],autolag='AIC')
    DF_OUTPUT=pd.Series(DF_TEST[0:4],index=['Test Statistics','P-value','Lags','Observations'])
    for key,value in DF_TEST[4].items():
        DF_OUTPUT['Critical Value (%s)'%key]=value
    print(DF_OUTPUT)
    if DF_OUTPUT['P-value'] < 0.05:
        print("Accept Null Hypothesis")
indexedDataset.dropna(inplace=True)
dickey(Dataset=indexedDataset,ts='case')   
from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(indexedDataset)
Trend=decompose.trend
Seasonal=decompose.seasonal
Residual=decompose.resid
plt.subplot(411)
plt.plot(indexedDataset,color='blue',label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(Trend,color='green',label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(Seasonal,color='brown',label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(Residual,color='Red',label='Residual')
plt.legend(loc='best')
plt.tight_layout()
decomposed_data=Residual
decomposed_data.dropna(inplace=True)
dickey(decomposed_data,ts='case')
#PACF
from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(indexedDataset,nlags=20)
lag_pacf=pacf(indexedDataset,nlags=20,method='ols')
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(indexedDataset)),linestyle='--',color='grey')
plt.title('ACF Q value')
plt.subplot(122)
plt.plot(lag_pacf)
#from pandas.tools.plotting import autocorrelation_plot
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(indexedDataset)),linestyle='--',color='grey')
plt.title('PACF P value')
plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import AR
print('AR')
model=ARIMA(indexedDataset,order=(7,0,2))
AR=model.fit(disp=-1)
plt.subplot(211)
plt.plot(indexedDataset,color='blue')
plt.plot(AR.fittedvalues,color='black')
#sum1=AR.fittedvalues
#sum2=indexedDataset['case']
#sum3=sum((sum1-sum2)**2)
plt.title('RSS: %.4f'% sum((AR.fittedvalues-indexedDataset['case'])**2))
print('MV.')
model2=ARIMA(indexedDataset,order=(7,0,0))
MV=model2.fit(disp=-1)
plt.subplot(221)
plt.plot(indexedDataset,color='blue')
plt.plot(MV.fittedvalues,color='black')
plt.title('Rss:%4f'% sum((MV.fittedvalues-indexedDataset['case'])**2))
#indexedDataset_log._infer_freq(indexedDataset_log['Date'])
model_ARIMA=ARIMA(indexedDataset,order=(7,0,2))
ARIMA=model_ARIMA.fit(disp=-1)
plt.subplot(221)
plt.plot(indexedDataset,color='blue')
plt.plot(ARIMA.fittedvalues,color='black')
plt.title('Rss:%4f'% sum((ARIMA.fittedvalues-indexedDataset['case'])**2))
#convert fitted values into series 
predict_ARIMA_diff=pd.Series(ARIMA.fittedvalues,copy=True)
print(predict_ARIMA_diff.head())
cumsum_predictions=predict_ARIMA_diff.cumsum()
print(cumsum_predictions.head())
predictions_ARIMA=pd.Series(indexedDataset['case'].iloc[0],index=indexedDataset.index)   
predictions_ARIMA=pd.Series(predictions_ARIMA.add(cumsum_predictions,fill_value=0))   
predictions_ARIMA.head()
#predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset,color='blue')
plt.plot(predictions_ARIMA,color='red')
indexedDataset
ARIMA.plot_predict(1,312)
x=ARIMA.forecast(steps=120)
## change in xlabel



















