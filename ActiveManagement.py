# --------------- ACTIVE PORTFOLIO MANAGEMENT --------------
# Project Completed for APPM 4720 - Open Topics in Applied Math - Optim. in Finance
# Date: 12/12/24

# Library Install
import yfinance as yf
import pandas as pd
import numpy as np
import csv

def fetch_and_filter_data(tickers):
    selected_tickers = []
    historical_data = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get("marketCap", 0)
            avg_volume = stock.info.get("averageVolume", 0)
            current_price = stock.info.get("currentPrice", 0)

            if((market_cap >= 2e6) & (avg_volume*current_price >= 20e6)):
                data = stock.history(period="5y")
                print("\t" + ticker + " --> Successful")
                if(not data.empty):
                    selected_tickers.append(ticker)
                    historical_data[ticker] = data["Close"]

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return selected_tickers, pd.DataFrame(historical_data)


def main():
    candidate_tickers = []

    file_path = input("Filepath for Tickers: ")
    with open(file_path, 'r', encoding='utf-8-sig') as data:
        for line in csv.reader(data):
            candidate_tickers.append(line[0])

    print("* Loading Candidate Tickers...")
    print("* Candidate Tickers:")
    for i in range(len(candidate_tickers)):
        print("\t" + candidate_tickers[i])

    print("* Fetching and Filtering Historical Data...")
    selected_tickers, price_data = fetch_and_filter_data(candidate_tickers)

    print("* Selected Tickers:")
    for i in range(len(selected_tickers)):
        print("\t" + selected_tickers[i])

    print("* Validating Historical Data...")
    price_data.index = pd.to_datetime(price_data.index, format='%Y-%m-%d')
    price_data = price_data.dropna(axis=1)
    print("* Validation Complete")

    print("* Segregating Data into Calibration and Test Data...")
    n = len(price_data)
    calibration_data = price_data.iloc[:int(n*0.6)]
    test_data = price_data.iloc[int(n*0.6):]
    print("* Segregation Complete")

    print("* Saving Datasets...")
    calibration_data.to_csv("Calibration_Data.csv")
    test_data.to_csv("Test_Data.csv")
    print("* Saving Datasets Complete")
    

    print("------------------------- End of Program -------------------------")
    return -1

if __name__ == '__main__':
    main()










