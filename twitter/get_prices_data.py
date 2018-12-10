import json
import numpy as np
import os
import pandas as pd
import urllib.request

class GetPricesData:
    def JSONDictToDF(self, d):
        '''
        Converts a dictionary created from json.loads to a pandas dataframe
        d:      The dictionary
        '''
        n = len(d)
        cols = []
        if n > 0:  # Place the column in sorted order
            cols = sorted(list(d[0].keys()))
        df = pd.DataFrame(columns=cols, index=range(n))
        for i in range(n):
            for coli in cols:
                df.set_value(i, coli, d[i][coli])
        return df


    def GetAPIUrl(self, cur):
        '''
        Makes a URL for querying historical prices of a cyrpto from Poloniex
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
        '''

        u = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' + cur + '&start=1520985600&end=9999999999&period=300'
        return u


    def GetCurDF(self, cur, fp):
        '''
        cur:    3 letter abbreviation for cryptocurrency (BTC, LTC, etc)
        fp:     File path (to save price data to CSV)
        '''
        print("called getcurdf function")
        openUrl = urllib.request.urlopen(GetPricesData.GetAPIUrl(self, cur))
        r = openUrl.read()
        openUrl.close()

        d = json.loads(r.decode())
        df = GetPricesData.JSONDictToDF(self, d)
        df.to_csv(fp, sep=',')
        print("returning df")
        return df

    def main(self):
        # %%Path to store cached currency data
        datPath = 'data/'
        if not os.path.exists(datPath):
            os.mkdir(datPath)
        # Different cryptocurrency types
        cl = ['BTC']  # 'LTC', 'ETH', 'XMR'
        # Columns of price data to use
        CN = ['close', 'high', 'low', 'open', 'volume']
        # Store data frames for each of above types
        D = []
        for ci in cl:
            print("iterating for ", ci)
            dfp = os.path.join(datPath, ci + '.csv')
            try:
                df = pd.read_csv(dfp, sep=',')
            except FileNotFoundError:
                df = GetPricesData.GetCurDF(self, ci, dfp)
            # D.append(df)
        print("D: ", D)
