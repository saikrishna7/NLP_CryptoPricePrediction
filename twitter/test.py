import pandas as pd
import sys
import datetime

def main(is_testing_set):
    test_data_prices = pd.read_csv("data/BTC.csv")
    if is_testing_set:
        test_data_prices = pd.read_csv("data/BTC.csv")
        test_data_prices.drop(columns=['Unnamed: 0'], inplace=True)
        test_data_prices["datetime"] = pd.to_datetime(test_data_prices['date'], unit='s')
        test_data_prices = test_data_prices[test_data_prices.datetime >= datetime.date(2018, 4, 17)]
        test_data_prices.to_csv("data/test_data_prices.csv", index=False)
        test_data_prices.reset_index(inplace=True, drop=True)

    # april 11, 12 am: 1523404800
    # March 25, 2018 12:00:00 AM : 1521936000
    # March 14, 2018 12:00:00 AM : 1520985600
    # April 17, 2018 12:00:00 AM : 1523923200

    prices = pd.read_csv("data/BTC.csv")
    # prices = prices[prices.date < 1523404800]
    prices = prices[prices.date >= 1520985600]
    prices = prices[prices.date < 1523923200]
    prices.drop(columns=['Unnamed: 0'], inplace=True)
    prices["datetime"] = pd.to_datetime(prices['date'], unit='s')
    prices.to_csv("data/train_data_prices.csv", index=False)


if __name__ == "__main__":
   main(sys.argv[1])
