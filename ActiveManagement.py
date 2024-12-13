# --------------- ACTIVE PORTFOLIO MANAGEMENT --------------
# Project Completed for APPM 4720 - Open Topics in Applied Math - Optim. in Finance
# Date: 12/12/24

# Library Install
import yfinance as yf
import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression

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

def fetch_and_filter_benchmark():
    sp500 = yf.Ticker("^GSPC")
    try:
        sp500_data = sp500.history(period='5y')["Close"]
        return pd.DataFrame(sp500_data)
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
    return -1

def covariance_matrix_build(calibration_data):
    returns = calibration_data.pct_change().dropna()
    covariance_matrix = returns.cov()
    return covariance_matrix

def beta_estimator(spx500_calibration_data, calibration_data):
    temp_spx_returns = spx500_calibration_data.pct_change().dropna()
    temp_spx_returns = temp_spx_returns.to_numpy()
    returns = calibration_data.pct_change().dropna()
    betas = {}

    spx_returns = []
    for i in range(len(temp_spx_returns)):
        spx_returns.append(float(temp_spx_returns[i][0]))

    for stock in returns.columns:
        model = LinearRegression()
        model.fit(np.array(spx_returns).reshape(-1,1), returns[stock].values)
        betas[stock] = float(model.coef_[0])
    return betas

def momentum_calculator(calibration_data):
    returns = calibration_data.pct_change().dropna()
    alphas = {}

    momentum = returns[-60:].mean()
    momentum_rank = momentum.rank(ascending=False)
    alpha = momentum_rank/len(momentum_rank)
    alphas = alpha.to_dict()
    return alphas

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
    price_data.to_csv("Price_Data.csv")
    calibration_data.to_csv("Calibration_Data.csv")
    test_data.to_csv("Test_Data.csv")
    print("* Datasets Saved")

    print("* Fetching Benchmark Data...")
    sp500_data = fetch_and_filter_benchmark()

    print("* Segregating Benchmark Data...")
    m = len(sp500_data)
    sp500_calibration_data = sp500_data.iloc[:int(m*0.6)]
    sp500_test_data = sp500_data.iloc[int(m*0.6):]
    print("* Segregation Complete")

    print("* Saving Benchmark Dataset...")
    sp500_data.to_csv("Benchmark_Data.csv")
    sp500_calibration_data.to_csv("Benchmark_Calibration_Data.csv")
    sp500_test_data.to_csv("Benchmark_Test_Data.csv")
    print("* Benchmark Data Saved")

    print("* Building Covariance Matrix From Calibration Data...")
    covariance_matrix = covariance_matrix_build(calibration_data)
    print(covariance_matrix.to_string())
    print("* Covariance Matrix Constructed")

    print("* Estimating Betas...")
    betas = beta_estimator(sp500_calibration_data, calibration_data)
    for key, value in betas.items():
        print("\t", key, ": ", value)
    print("* Beta Estimation Complete")

    print("* Calculating Alphas...")
    alphas = momentum_calculator(calibration_data)
    for key, value in alphas.items():
        print("\t", key, ": ", value)
    print("* Alpha Estimation Complete")

    print("* Building Long Only Optimization...")
    '''
    print("* Long Only Optimization Complete")

    print("* Builing Long-Short Optimization...")
    print("* Long-Short Optimization Complete")

    print("* Builing 130/30 Optimization...")
    print("* 130/30 Optimization Complete")

    print("* Calculating Portfolio Restults...")
    print("* Portfolio Results Calculated")

    print("* Visualizing Results...")
    print("* Visualizations Saved")

    print("* Performing Rolling-Window Calculations...")
    print("* Rolling-Window Calculations Complete")
    '''
    

    print("------------------------- End of Program -------------------------")
    return -1

if __name__ == '__main__':
    main()










