import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn
import talib
import talib.abstract as tabs

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
tickets = first_table['Symbol'].values.tolist()
tickets.remove('BRK.B')
tickets.remove('BF.B')

all_stocks = []
for ticket in tickets:
    stocks = pd.read_csv('./datasets/yahoo/' + ticket + '.csv')
    stocks.rename(columns=str.lower, inplace=True)
    all_stocks.append(stocks)

def feature_extraction(df):
    
    # ROC
    roc = tabs.ROC(df, timeperiod=1)
    roc = np.nan_to_num(roc)
    df['roc'] = roc
    
    # SMA 10
    sma = tabs.SMA(df, timeperiod=10)
    sma = np.nan_to_num(sma)
    df['sma'] = sma
    
    # MACD, MACD SIGNAL and MACD HIST
    
    macd, macdsignal, macdhist = talib.MACD(df['close'])
    macd = np.nan_to_num(macd)
    macdsignal = np.nan_to_num(macdsignal)
    macdhist = np.nan_to_num(macdhist)
    df['macd'] = macd
    df['macd_signal'] = macdsignal
    df['macd_hist'] = macdhist
    
    # CCI 24
    cci = tabs.CCI(df, timeperiod=24)
    cci = np.nan_to_num(cci)
    df['cci'] = cci
    
    #     MTM 10 
    mtm = tabs.MOM(df, timeperiod=10)
    mtm = np.nan_to_num(mtm)
    df['mtm'] = mtm
    
    #     RSI 5 
    rsi = tabs.RSI(df, timeperiod=5)
    rsi = np.nan_to_num(rsi)
    df['rsi'] = rsi
    
    #     WNR 9
    wnr = tabs.WMA(df, timeperiod=9)
    wnr = np.nan_to_num(wnr)
    df['wnr'] = wnr
    
    #     SLOWK & SLOWD
    slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
    slowk = np.nan_to_num(slowk)
    slowd = np.nan_to_num(slowd)
    df['slowk'] = slowk
    df['slowd'] = slowd
    
    #     ADOSC 
    adosc = tabs.ADOSC(df)
    adosc = np.nan_to_num(adosc)
    df['adosc'] = adosc
    
    #     AARON
    aroondown, aroonup = talib.AROON(df['high'], df['low'])
    aroondown = np.nan_to_num(aroondown)
    aroonup = np.nan_to_num(aroonup)
    df['aroon_down'] = aroondown
    df['aroon_up'] = aroonup
    
    #     BBANDS
    upper, middle, lower = talib.BBANDS(df['close'], matype=0)
    upper = np.nan_to_num(upper)
    df['upper'] = upper
    middle = np.nan_to_num(middle)
    df['middle'] = middle
    lower = np.nan_to_num(lower)
    df['bbands'] = lower

def feature_normalization(df):
    features = ['volume', 'sma', 'rsi', 'wnr', 'slowk', 'slowd', 'adosc']
    scaler = MinMaxScaler()
    for f in features:
        damn = np.array(df[f]).reshape((-1, 1))
        df[f + '_mm'] = scaler.fit_transform(damn).reshape((-1))


for ticket, stock in zip(tickets, all_stocks):
    s = stock.copy()
    feature_extraction(s)
    feature_normalization(s)
    s.to_csv('./datasets/enriched/' + ticket + '.csv', index=False)