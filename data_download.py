import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from tqdm import tqdm

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    df=pd.read_html(url,storage_options=headers)[0]
    tickers=df['Symbol'].str.replace('.','-',regex=False).tolist()
    return tickers

def fetch_historical_data(tickers,cache_file='sp500_raw_data.pkl'):
    if os.path.exists(cache_file):
        print(f"Loading Data From: {cache_file}")
        return pd.read_pickle(cache_file)
    print(f"Downloading Data: {tickers}")
    data = yf.download(tickers, period="10y", interval="1d", threads=True, auto_adjust=True)
    print(f"Saving Data To: {cache_file}")
    data.to_pickle(cache_file)
    return data

def prep_raw_indicators(df):
    mean_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['Z_raw'] = (df['close'] - mean_20) / std_20
    sma_50 = df['close'].rolling(window=50).mean()
    df['SMA_raw'] = (df['close'] - sma_50) / sma_50
    high_low_diff = df['high'] - df['low']
    high_low_diff = high_low_diff.replace(0,1e-8)
    money_flow_mult = ((df['close']-df['low']) - (df['high'] - df['low']))/high_low_diff
    volume_sum = df['volume'].rolling(window=20).sum().replace(0,1e-8)
    money_flow_vol=money_flow_mult*df['volume']
    df['CMF_raw'] = money_flow_vol.rolling(window=20).sum()/volume_sum
    df=df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=['Z_raw', 'SMA_raw', 'CMF_raw'])
def calculate_orthogonal_weights(df):
    halflife = 504
    weights = np.exp(np.log(0.5) / halflife*np.arange(len(df))[::-1])
    z=df['Z_raw'].values
    c=df['CMF_raw'].values
    s=df['SMA_raw'].values
    results={}
    results['sigma_Z']=float(np.std(z))
    z_norm=z/results['sigma_Z']
    x_c=sm.add_constant(z_norm)
    model_c=sm.WLS(c,x_c,weights=weights).fit()
    results['alpha_C'] = float(model_c.params[0])
    results['beta_C_Z'] = float(model_c.params[1])
    c_res=model_c.resid
    results['Sigma_C_res'] = float(np.std(c_res))
    c_norm=c_res/results['Sigma_C_res']
    x_s=sm.add_constant(np.column_stack((z_norm,c_norm)))
    model_s=sm.WLS(s,x_s,weights=weights).fit()
    results['alpha_S'] = float(model_s.params[0])
    results['beta_S_Z'] = float(model_s.params[1])
    results['beta_S_C'] = float(model_s.params[2])
    s_res=model_s.resid
    results['sigma_S_res'] = float(np.std(s_res))
    return results
def run_alpha_factory():
    tickers = get_sp500_tickers()
    data = fetch_historical_data(tickers)
    master_weights={}
    for ticker in tqdm(tickers, desc='Regressing'):
        try:
            if('Close', ticker) not in data.columns:
                continue
            df_ticker=pd.DataFrame({
                'close': data['Close'][ticker],
                'high': data['High'][ticker],
                'low': data['Low'][ticker],
                'volume': data['Volume'][ticker],
            }).dropna()
            if len(df_ticker)<500:
                continue
            df_prepped=prep_raw_indicators(df_ticker)
            ticker_weights=calculate_orthogonal_weights(df_prepped)
            master_weights[ticker]=ticker_weights
        except Exception as e:
            print(f"Error: {e}")
            continue
    output_file='factor_weights.json'
    with open(output_file, 'w') as f:
        json.dump(master_weights, f,indent=4)
    print(f"Generated {output_file} for {len(master_weights)} tickers")

if __name__ == "__main__":
    run_alpha_factory()