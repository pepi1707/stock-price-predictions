# This script scrapes the names of the companies from the S&P 500 index
# from wikipedia, then uses pandas datareader to scrape from yahoo
# the stock prices for each company for NUM_YEARS amount of time
# The time is divided per days
# Then saves the information for each company in a different csv file
# in the datasets/yahoo folder

import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
from datetime import datetime

NUM_YEARS = 2

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
tickets = first_table['Symbol'].values.tolist()

tickets.remove('BRK.B')
tickets.remove('BF.B')

end = datetime.now()
start = datetime(end.year-NUM_YEARS, end.month, end.day)

all_stocks = []

for ticket in tickets:
    x = DataReader(ticket, 'yahoo', start, end)
    print(len(all_stocks))
    all_stocks.append(x)

for ticket, stock in zip(tickets, all_stocks):
    stock.to_csv('./datasets/yahoo/' + ticket + '.csv', index=False)