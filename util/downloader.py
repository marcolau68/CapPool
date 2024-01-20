import yfinance as yf
import numpy as np
import pandas as pd
import sys
import bs4 as bs
import requests


START_DATE = "2000-01-01"
END_DATE = "2023-12-31"


def download_tickers():
    # Scrape Wikipedia page for the constituents
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'id': 'constituents'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    tickers = [s.replace('\n', '') for s in tickers]
    
    return tickers

def download_ticker_data(ticker):
    # Download ticker data
    df = yf.download(ticker, start=START_DATE, end=END_DATE)

    # Drop the listed columns
    df = df.drop(["Open", "High", "Low", "Volume", "Close"], axis = 1)

    # Use adjusted close as close, account for dividend and stock splits
    df = df.rename(columns={"Adj Close": "Close"})

    # Find Log Returns
    df["log_returns"] = np.emath.log(df.Close / df.Close.shift(1))

    # Write data to csv and save to raw_data folder
    path = "raw_data/" + ticker + ".csv"
    df.to_csv(path)

    print(f"{ticker} download complete")

    return None

def download_constituents_data():
    tickers = download_tickers()

    for ticker in tickers:
        download_ticker_data(ticker)
    
    return None


# download_ticker_data("MMM")
download_constituents_data()
# print("Test")