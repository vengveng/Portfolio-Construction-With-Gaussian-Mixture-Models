import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

from optimization_tools import (
    settings, KDE, GMM, Moments, Portfolio,
    iteration_depth,  portfolio_evaluation, get_annualization_factor, 
    prepare_returns, colorize, preload_bandwidth_csv, update_portfolio)


# Configuration block of a single configuration
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
indices = ['FTSE100', 'HSI', 'S&P500', 'STOXX50']
index_name = indices[2] # THIS ALLOWS TO CHANGE THE INDEX
N = 100
GAMMA = 10
random_seed = 1

# Configuration settings
config = {
    'limit_year': None,
    'data_frequency': "weekly",
    'rebalancing_frequency': "annual",
    'master_index': None,
    'global_tickers': None,
    'timeout': 60,
    'window_size': 1,
    'window_unit': 'years',
}

# Static portfolio configurations
portfolio_configs = [
    {"name": "equal"},
    {"name": "value"},
    {"name": "min_var"},
    {"name": "markowitz", "args": {"gamma": 1 / GAMMA}},
    {"name": "max_sharpe"},
    # {"name": "erc", 'solver': 'ipopt'},
    {"name": "kde", "prefit_kde": True, "args": {"gamma": GAMMA}},
    {"name": "gmm", "prefit_gmm": True, "args": {"gamma": GAMMA}},
    {"name": "kde_max_sharpe", "prefit_kde": True},
    {"name": "gmm_max_sharpe", "prefit_gmm": True},
    ]

settings.update_settings(**config)
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------


# Paths
tqdm.write(colorize("Loading data...", "blue"))
root = os.path.dirname(__file__)
equity_path = os.path.join(root, 'data', 'TRI_cleaned.csv')
cap_path = os.path.join(root, 'data', 'MV_cleaned.csv')
benchmarks_path = os.path.join(root, 'data', 'indices.csv')
history_path = os.path.join(root, 'data', 'index_histories', f'{index_name}_HISTORY.csv')

# Load
all_prices = pd.read_csv(equity_path, index_col=0, parse_dates=True)
all_caps = pd.read_csv(cap_path, index_col=0, parse_dates=True)
benchmark_df = pd.read_csv(benchmarks_path, index_col=0, parse_dates=True, dayfirst=True)

# Process
all_returns = prepare_returns(all_prices, frequency=settings.data_frequency)
masterIndex = all_returns.index
tickers = all_returns.columns
annualization_factor = get_annualization_factor(masterIndex)
settings.update_settings(ANNUALIZATION_FACTOR=annualization_factor, master_index=masterIndex, index_name=index_name, global_tickers=list(tickers))

bandwidth_df = preload_bandwidth_csv(std=True)
(lambda df: globals().update({"bandwidth_df": df}))(bandwidth_df) # necessary for multiprocessing

valid_tickers = set(tickers)
index_history = pd.read_csv(history_path, parse_dates=True, index_col=0)
index_history['TICKERS'] = index_history['TICKERS'].apply(lambda x: x.split(','))

all_caps = all_caps.resample('D').ffill()
all_caps = all_caps.reindex(masterIndex, method='bfill')
all_caps = all_caps.reindex(columns=tickers, fill_value=0)
all_caps.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

rf_series = benchmark_df['3mYIELD'].sort_index()
rf_series = rf_series / (100 * settings.ANNUALIZATION_FACTOR)
rf_rate = rf_series.reindex(masterIndex, method='ffill')

# Trackers
portfolio_returns = pd.DataFrame(index=masterIndex, columns=[p["name"] for p in portfolio_configs])

previous_tickers = None
dates = pd.Series(index=masterIndex)
moments = {}

indexIterator = iteration_depth(window=settings.window_size, window_units=settings.window_unit)
progress = tqdm(indexIterator, desc=colorize('Computing moments', 'yellow'), unit='step')
start_time = time.time()

# Moment precomputation
#--------------------------------------------------------------------------------
for step in progress:

    optimizationIndex = indexIterator[step]['optimizationIndex']
    evaluationIndex   = indexIterator[step]['evaluationIndex']
    lastDate = dates.loc[optimizationIndex].index[-1]

    if len(evaluationIndex) == 0:
        break

    # Setting True here will use the maximum number of valid tickers in the index history
    # ---------------------------------------------------------------------------------------------------------
    # indices = update_portfolio(index_history, lastDate, N, random_seed, previous_tickers, True, valid_tickers)
    indices = update_portfolio(index_history, lastDate, N, random_seed, previous_tickers, False, valid_tickers)
    # ---------------------------------------------------------------------------------------------------------

    sampleRf = rf_rate.loc[optimizationIndex].iloc[-1]
    sampleReturns     = all_returns.loc[optimizationIndex, indices].sort_index(axis=1)
    evaluationReturns = all_returns.loc[evaluationIndex, indices].sort_index(axis=1)
    sampleCaps = all_caps.loc[optimizationIndex, indices].sort_index(axis=1).iloc[-1]

    zero_cols = sampleReturns.columns[(sampleReturns == 0).all()]
    if not zero_cols.empty:
        sampleReturns = sampleReturns.drop(columns=zero_cols)
        evaluationReturns = evaluationReturns.drop(columns=zero_cols)
        sampleCaps = sampleCaps.drop(labels=zero_cols)
        indices = [t for t in indices if t not in zero_cols]

    start = dates.loc[optimizationIndex].index[0].date()
    end = lastDate.date()
    eval_start = dates.loc[evaluationIndex].index[0].date()

    progress.set_postfix({
        "Start": str(start),
        "End": str(end),
        "Eval": str(eval_start),
        "Sample": sampleReturns.shape[0],
        "Assets": sampleReturns.shape[1]})

    mom = Moments(sampleReturns, evaluationReturns, sampleRf, sampleCaps)
    kde = KDE(sampleReturns, matrix='diagonal', std=True, bandwidth_df=bandwidth_df, lock=True)
    gmm = GMM(sampleReturns, k=8, lock=True)
    moments[step] = mom, kde, gmm


# Optimization
#--------------------------------------------------------------------------------
def evaluate_portfolios(args):
    step, mom, kde, gmm, configs = args
    result_returns = {}
    result_weights = {}

    for config in configs:
        kwargs = {
            "returns": None,
            "type": config["name"],
            "risk_free_rate": None,
            "moments_instance": mom,
            "prefit_kde": kde if config.get("prefit_kde") else None,
            "prefit_gmm": gmm if config.get("prefit_gmm") else None,
            "args": config.get("args", {}),
            "solver": config.get("solver", None),
            "verbose": True
        }

        portfolio = Portfolio(**kwargs)
        result_returns[config["name"]] = portfolio.evaluate_performance(mom.eval_returns).values
        result_weights[config["name"]] = (mom.eval_index, portfolio.ticker, portfolio.actual_weights)

    return step, result_returns, result_weights

jobs = [(step, mom, kde, gmm, portfolio_configs) for step, (mom, kde, gmm) in moments.items()]

with Pool() as pool:
    description = colorize("Evaluating portfolios", "yellow")
    results = list(tqdm(pool.imap(evaluate_portfolios, jobs), total=len(jobs), desc=description))

# Evaluation
for step, returns_dict, weights_dict in results:
    for name, values in returns_dict.items():
        portfolio_returns.loc[moments[step][0].eval_index, name] = values

tqdm.write(colorize("Saving data...", 'blue'))
# portfolio_returns.to_csv(os.path.join(root, 'data', 'portfolio_returns','portfolio_returns.csv'))
tqdm.write(colorize("Done!", 'green'))

print((1 + portfolio_returns[portfolio_returns.index.year <= 2025]).cumprod().tail(1))
print(portfolio_evaluation(portfolio_returns, rf_rate)['SR'])
print(f"Optimization Runtime: {(time.time() - start_time):2f}s")


# Weekly, annual, 1 year, 100
#                equal     value   min_var markowitz max_sharpe       kde       gmm kde_max_sharpe gmm_max_sharpe
# DATE                                                                                                           
# 2025-03-23  6.211003  5.667261  5.435348  4.731493   4.013137  5.716828  3.582871        3.76737        2.95561
# equal             0.528038
# value             0.524937
# min_var           0.589012
# markowitz         0.415776
# max_sharpe         0.42501
# kde               0.590509
# gmm               0.386481
# kde_max_sharpe    0.405084
# gmm_max_sharpe    0.326276
# dtype: object
# Optimization Runtime: 1.144427s