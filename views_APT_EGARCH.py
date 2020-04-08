# -*- coding: utf-8 -*-
"""
Created on Mon Apr 6 20:56:43 2019

@author: qy
"""
# NOT USING MONTHLY DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
#import statsmodels.api as sm
from sklearn.decomposition import PCA
import subprocess
import time


def get_annual_ret_asset(start_date, end_date, tickers_list):
    '''Get annual return series for specified tickers, start and end date.
    Input:
        start_date: str; series start date, in 'yyyy-mm-dd' format
        end_date: str; series end date, in 'yyyy-mm-dd' format
        tickers_list: list of str; the specified tickers
    Output:
        ass_return: df; annual return series with dates as index, symbols as columns
    '''
    # get the asset adjusted close prices
    asset = web.DataReader(tickers_list, 'yahoo', start_date, end_date)
    ass_close = asset['Adj Close'].fillna(method='bfill')
    # calc the daily log returns
    ass_return = np.log(ass_close / ass_close.shift(1)).fillna(method='bfill')
    ass_return = ass_return[1:]
    # convert to annual returns
    ass_return *= 252
    return ass_return
    

def get_annual_ret_rf(start_date, end_date, ticker='^IRX'):
    '''Get annual return series for risk-free asset, start and end date.
    Input:
        start_date: str; series start date, in 'yyyy-mm-dd' format
        end_date: str; series end date, in 'yyyy-mm-dd' format
        ticker: str; the ticker of risk-free rate, default as '^IRX': 13-Week Treasury Bill
    Output:
        risk_free: pd.Series; annual risk-free return series with dates as index
    '''
    # get the risk-free prices
    RF_quotes = web.DataReader(ticker, 'yahoo', start_date, end_date)['Close']
    # get the expected risk-free rate
    risk_free = np.mean(1/(1-RF_quotes*0.01)-1)
    return risk_free
    

def get_annual_ret_mkt(start_date, end_date, ticker='^GSPC'):
    '''Get annual return series for risk-free asset, start and end date.
    Input:
        start_date: str; series start date, in 'yyyy-mm-dd' format
        end_date: str; series end date, in 'yyyy-mm-dd' format
        ticker: str; the ticker of risk-free rate, default as '^GSPC': SP500
    Output:
        mkt_return: pd.Series; annual market return series with dates as index
    '''
    # get the asset adjusted close price
    market = web.DataReader('^GSPC', 'yahoo', start_date, end_date)
    market_close = market['Adj Close'].fillna(method='bfill')
    # calc the daily log returns
    mkt_return = np.log(market_close / market_close.shift(1)).fillna(method='bfill')
    mkt_return = mkt_return[1:]
    # convert to annual returns
    mkt_return *= 252
    return mkt_return
    

def prep_returns_data(tickers, start_time, end_time):
    # Prepare returns data for assets, risk-free, and market
    # load data
    ass_ret = get_annual_ret_asset(start_time, end_time, tickers)
    mkt_ret = get_annual_ret_mkt(start_time, end_time)
    rf_ret = get_annual_ret_rf(start_time, end_time)
    print('*** Done: load data ***')
    
    # Adjusting asset returns to monthly frequency by taking the averages
    ass_ret_byMon, mkt_ret_byMon = ass_ret, mkt_ret
    #ass_ret_byMon = ass_ret.groupby(pd.Grouper(freq='MS')).mean()
    #mkt_ret_byMon = mkt_ret.groupby(pd.Grouper(freq='MS')).mean()
    # Output asset returns data
    ass_ret_byMon.to_csv('asset.csv', header=True)
    mkt_ret_byMon.to_csv('market.csv', header=True)
    return rf_ret, ass_ret_byMon, mkt_ret_byMon
    

def prep_factors_data(ass_ret_byMon):
    # Load original factor values I_j's
    factors = pd.read_csv(r"./multi-factors.csv", index_col=0)
    factors.index = ass_ret_byMon.index
    factors = factors.drop(columns='RF', errors='ignore')
    print("Number of factors to select from: %d" % len(factors.columns))
    
    # standardize the factors
    standard_factors = (factors - factors.mean())/factors.std()
    
    # calc the covariance matrix of factors
    factors_cov = standard_factors.cov()
    factors_corr = standard_factors.corr()
    # plot the correlation matrix
    fig1, ax1 = plt.subplots(figsize=(12,10))
    sns.heatmap(factors_corr, annot=True, linewidths=0.05, linecolor='white', 
                annot_kws={'size':8,'weight':'bold'}, ax=ax1)
    
    print('*** Done: load factors ***')
    return standard_factors, factors_cov


def factors_PCA(standard_factors):
    # Conduct PCA to determine factors
    pca = PCA(n_components=0.95)   # to explain 95% variations
    factors_pca_mat = pca.fit_transform(standard_factors.values)
    num_of_selected = factors_pca_mat.shape[1]
    print("Number of factors after PCA: %d" % num_of_selected)
    # adjust the column names
    column_names = ['f%d' %(i+1) for i in range(num_of_selected)]
    factors_pca = pd.DataFrame(factors_pca_mat, columns=column_names)
    factors_pca.index = standard_factors.index
    
    # calc the covariance matrix of factors after PCA
    factors_pca_cov = factors_pca_df.cov()
    factors_pca_corr = factors_pca_df.corr()  
    # plot the correlation matrix
    fig2, ax2 = plt.subplots(figsize=(12,10))
    sns.heatmap(factors_pca_corr, annot=True, linewidths=0.05, linecolor='white', 
                annot_kws={'size':8,'weight':'bold'}, ax=ax2)
    
    print('*** Done: PCA on factors ***')
    return factors_pca, factors_pca_cov
    

def gen_view_w_uncertainty_EGARCH(path2script):
    # Use EGARCH in R to derive investor views and uncertainty levels - may cost 8 min
    # define the command, file path and arguments
    start_t = time.time()
    command = 'Rscript'
    args = [str(rf)]
    # build subprocess command
    cmd = [command, path2script] + args
    # use check_call method to run the command with args
    subprocess.check_call(cmd)
    print('*** Done: Run R Script - time used: {:.2f} sec ***'.format(time.time() - start_t))
    

def check_pred_accuracy(fut_start_date, fut_end_date):
    R_fut_expected = pd.read_csv('pred.csv', index_col=0)['view']
    # Check prediction accuracy
    tickers = R_fut_expected.index.tolist()
    # load future data
    R_fut_actual = get_annual_ret_asset(fut_start_date, fut_end_date, tickers).mean()
    
    fig, ax = plt.subplots(figsize=(30,6))
    ax.scatter(tickers, R_fut_actual, color='black', marker='P', label='actual')
    ax.scatter(tickers, R_fut_expected, color='red', marker='o', label='EGARCH-M')
    ax.legend(loc=0, fontsize='x-large')
    plt.title('Expected Return Plot',fontsize=24)
    plt.show()
    MSE = np.mean((R_fut_expected-R_fut_actual)**2)
    return MSE
    

if __name__ == '__main__':
    asset_list = ['XRX','HES','CMG','XLNX','DVN','IPGP','AMD',
           'MSCI','HBI','NBL','CDNS','WDC','AAPL','CELG','APTV',
           'CAG','QCOM','LRCX','ULTA','WYNN','APA','NVDA','NFLX',
           'PVH','SNPS','ALGN','FB','KLAC','TDG','AMP','BBY','MU',
           'TSN','GPN','CPRT','FLT','GRMN','DOV','MAS','MCHP','DISH',
           'NSC','JEC','HP','MCO','FAST']
    start_time = '2013-01-01'
    end_time = '2018-12-31'
    
    # Prepare returns data for assets, risk-free, and market
    rf, ass_ret, mkt_ret = prep_returns_data(asset_list, start_time, end_time)
    # Load original factor values I_j's
    factors_raw_df, factors_cov_raw = prep_factors_data(ass_ret)
    # Conduct PCA to determine factors
    factors_pca_df, factors_cov_pca = factors_PCA(factors_raw_df)
    # Output factors data
    factors_pca_df.to_csv('factors.csv', header=True)
    
    # Use EGARCH-M in R to derive investor views and uncertainty levels - may cost 8 min
    rscript_path = 'C:\\_Programming\\GitHubRepo\\BL_Allocation_APT_EGARCH\\EGARCH.R' 
    gen_view_w_uncertainty_EGARCH(rscript_path)
    
    # Check prediction accuracy
    fut_start_time = end_time
    fut_end_time = (pd.Timestamp(end_time) + pd.Timedelta(30, unit='D')).strftime('%Y-%m-%d')
    pred_MSE = check_pred_accuracy(fut_start_time, fut_end_time)
    
    