import csv
import numpy as np
import talib
import pandas as pd
import yfinance as yf

tickers = []
with open("Top_Score_Output.csv", "r") as ticker_list:
    reader = csv.reader(ticker_list)
    next(reader)
    for row in reader:
        tickers.append(row)

def fetch_data(ticker,start_date,end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.dropna(inplace=True)
    return df
    
def calculate_KAMA(price, period, pow1=2, pow2=30):
    change = np.abs(price.diff(period))
    volatility = price.diff().abs().rolling(period).sum()
    ER = change / volatility
    sc = ((ER*(2.0/(pow1+1)-2.0/(pow2+1)) + 2/(pow2+1)) ** 2).values
    kama = []
    for n in range(len(price)):
        if n < period:
            kama.append(np.nan)
        elif n == period:
            kama.append(price[n])
        else:
            kama.append(kama[-1] + sc[n] * (price[n] - kama[-1]))
    return pd.Series(kama, index=price.index)

def calculate_PSAR(high, low, acceleration=0.02, maximum=0.2):
    return talib.SAR(high, low, acceleration, maximum)

#1,2,3,5,8,13,21 from the list
select_tickers = [tickers[0], tickers[1], tickers[2], tickers[4], tickers[7], tickers[12], tickers[20]]

start_date = "2023-03-26"
end_date = "2023-06-26"
price_history = []
for ticker in select_tickers:
    try:
        data = fetch_data(ticker[0], start_date, end_date)
        KAMA = calculate_KAMA(data['Close'],10)
        PSAR = calculate_PSAR(data['High'], data['Low'])
        data['KAMA'] = KAMA
        data['PSAR'] = PSAR
        data['position'] = np.where(data['KAMA'] > data['PSAR'], 1, -1)  # 1 for long position, -1 for short
        # Calculate the daily return
        data['daily_return'] = data['Close'].pct_change() * data['position'].shift()
        # Calculate the cumulative return
        data['cumulative_return'] = (1 + data['daily_return']).cumprod()
        # Calculate the total profit/loss
        profit_loss = data['cumulative_return'].iloc[-1] - 1
        print(data)
        price_history.append(data)

        data.to_csv(f"tests/{ticker}_data.csv")
    except Exception as e:
        print(f"error loading {ticker}: {str(e)}")

