import os
import sys
import time
import random
import warnings
import threading
from copy import deepcopy
from itertools import cycle
from pathos.multiprocessing import Pool

import numpy as np
import pandas as pd
import cvxpy as cp
from cyipopt import minimize_ipopt
from scipy.special import softmax, logsumexp
from scipy.optimize import Bounds, LinearConstraint, minimize

from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning

from numba import njit
from tqdm import tqdm

from ctypes import CDLL, c_double, POINTER, c_size_t


warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.mixture._base")

lib = CDLL('lib/portfolio_lib.so')

lib.max_sharpe_objective_and_jacobian.argtypes = [
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    c_size_t
]
lib.max_sharpe_objective_and_jacobian.restype = c_double

class Settings:
    VALID_FREQUENCIES = ('monthly', 'annual', 'daily', 'weekly')
    VALID_MODES = ('fast')

    def __init__(self):
        self.limit_year = None
        self.data_frequency = 'monthly'
        self.rebalancing_frequency = 'annual'
        self.ANNUALIZATION_FACTOR = None
        self.master_index = None
        self.global_tickers = None
        self.mode = 'fast'
        self.gamma_linspace = np.linspace(-0.5, 1.5, 101)
        self.timeout = 15
        self.window_size = 5
        self.window_unit = 'year'
        self.index_name = None

        self.validate()

    def validate(self):
        if self.data_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"Invalid data frequency: {self.data_frequency}. Must be one of {self.VALID_FREQUENCIES}.")
        if self.rebalancing_frequency not in self.VALID_FREQUENCIES:
            raise ValueError(f"Invalid rebalancing frequency: {self.rebalancing_frequency}. Must be one of {self.VALID_FREQUENCIES}.")
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {self.VALID_MODES}.")

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate()

settings = Settings()

class Spinner:
    def __init__(self, message="Processing...", color="white"):
        self.spinner = cycle(['|', '/', '-', '\\'])
        self.stop_running = threading.Event()
        self.message_text = message
        self.lock = threading.Lock()  # To prevent conflicts with message updates
        self.color_code = {
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "white": "\033[37m",
            "reset": "\033[0m"
        }
        self.current_color = color 

    def start(self):
        def run_spinner():
            sys.stdout.write(self.message_text + " ")
            while not self.stop_running.is_set():
                with self.lock:
                    colored_symbol = self.color_code.get(self.current_color, self.color_code["white"]) + next(self.spinner) + self.color_code["reset"]
                    sys.stdout.write(colored_symbol)  
                    sys.stdout.flush()
                    sys.stdout.write('\b')
                time.sleep(0.1)

        self.thread = threading.Thread(target=run_spinner)
        self.thread.start()

    def stop(self):
        self.stop_running.set()
        self.thread.join()

    def message(self, new_message, color="white"):
        """Update the status message and color while the spinner is running."""
        with self.lock:
            sys.stdout.write('\b \b')  
            sys.stdout.flush()
            self.current_color = color
            colored_message = self.color_code.get(color, self.color_code["white"]) + new_message + self.color_code["reset"]
            sys.stdout.write('\r' + colored_message + " ")
            sys.stdout.flush()
            time.sleep(0.1)
            self.message_text = new_message

    def erase(self):
        """Erase the current message from the terminal."""
        with self.lock:
            sys.stdout.write('\r')
            sys.stdout.write(' ' * (len(self.message_text) + 2))
            sys.stdout.write('\r')
            sys.stdout.flush()
            self.message_text = ""

def annualized_mean(sample_mean: float) -> float:
    return (1 + sample_mean) ** settings.ANNUALIZATION_FACTOR - 1

def annualized_volatility(sample_std: float) -> float:
    return sample_std * np.sqrt(settings.ANNUALIZATION_FACTOR)

def sharpe_ratio(mean: float, volatility: float) -> float:
    if isinstance(volatility, pd.Series) and volatility.eq(0).any():
        return 0
    return mean / volatility

def portfolio_evaluation(monthlyReturns: pd.Series | np.ndarray, monthlyRFrate: pd.Series) -> dict:

    '''
    Evaluates the performance of a portfolio given its monthly returns. 
    It calculates and returns a dictionary containing the annualized mean return,
    annualized volatility, Sharpe ratio, minimum return, and maximum return of the portfolio.
    monthlyRFrate must be indexed by and be of same length as the sample of monthly returns 
    that is being evaluated.
    '''

    mean = monthlyReturns.mean()
    volatility = monthlyReturns.std()
    annualizedMean = annualized_mean(mean)
    annualizedVolatility = annualized_volatility(volatility)
    monthlyExcessReturn = monthlyReturns.sub(monthlyRFrate, axis=0)
    meanExcessReturn = monthlyExcessReturn.mean()
    annualizedExcessReturn = annualized_mean(meanExcessReturn)
    sharpeRatio = sharpe_ratio(annualizedExcessReturn, annualizedVolatility)
    minimum = monthlyReturns.min()
    maximum = monthlyReturns.max()

    portfolio_performance = {
        'mu': annualizedMean,
        'std': annualizedVolatility,
        'SR': sharpeRatio,
        'min': minimum,
        'max': maximum,
    }

    return portfolio_performance

def pperf(monthlyReturns: pd.Series | np.ndarray, monthlyRFrate: pd.Series) -> dict:

    mean = monthlyReturns.mean()
    volatility = monthlyReturns.std()
    annualizedMean = annualized_mean(mean)
    annualizedVolatility = annualized_volatility(volatility)
    monthlyExcessReturn = monthlyReturns.sub(monthlyRFrate, axis=0)
    meanExcessReturn = monthlyExcessReturn.mean()
    annualizedExcessReturn = annualized_mean(meanExcessReturn)
    sharpeRatio = sharpe_ratio(annualizedExcessReturn, annualizedVolatility)
    mdd = (1 + monthlyReturns).cummax().sub(1 + monthlyReturns).max()

    portfolio_performance = {
        'mu': annualizedMean,
        'std': annualizedVolatility,
        'SR': sharpeRatio,
        'CR': (1 + monthlyReturns).cumprod().iloc[-1] - 1,
        'VaR': monthlyReturns.quantile(0.05) * settings.ANNUALIZATION_FACTOR,
        'Skew': monthlyReturns.skew(),
        'MDD': mdd,
    }

    return pd.DataFrame(portfolio_performance)

class KDE():
    def __init__(
        self,
        returns: pd.DataFrame | pd.Series,
        h: float = None,
        matrix: str = 'isotropic',
        cv: int = 5,
        std: bool = False,
        bandwidth_df: pd.DataFrame = None,  # <- New optional argument
        lock: bool = None,
        **kwargs
    ):
        assert isinstance(cv, int) and cv > 1, "Invalid cross-validation parameter."

        self.returns = returns
        self.dim = len(returns.columns)
        self.kde_args = {'h': h, 'matrix': matrix}
        self.cv = cv
        self.std = std
        self.bandwidth_df = bandwidth_df
        self.loaded = False
        
        self.lock = lock
        if self.lock is not None:
            assert self.bandwidth_df is not None, (
                "When a lock is used, a valid bandwidth_df must be provided."
            )

        # TODO: Check if this is correct
        # self.bandwidth_matrix = self._construct_badnwidth_matrix() * settings.ANNUALIZATION_FACTOR
        self.bandwidth_matrix = self._construct_badnwidth_matrix()

    def _construct_badnwidth_matrix(self) -> np.ndarray:
        if self.kde_args['matrix'] == 'isotropic':
            if self.kde_args['h']:
                tqdm.write("Enforcing user isotropic bandwidth.")
                # print('Enforcing user isotropic bandwidth.')
                bandwidth_array = np.full(self.dim, self.kde_args['h'])
            else:
                bandwidth_array = self._isotropic_kde_bandwidth()
        else:
            if self.kde_args['h']:
                tqdm.write("Warning: User specified bandwidth will be ignored. 'h': {self.kde_args['h']} and 'matrix': 'diagonal' are mutually exclusive arguments.")
                # print(f"Warning: User specified bandwidth will be ignored. 'h': {self.kde_args['h']} and 'matrix': 'diagonal' are mutually exclusive arguments.")
            assert self.kde_args['matrix'] == 'diagonal', "Logical failure warning: Invalid bandwidth matrix specification."
            bandwidth_array = self._diagonal_kde_bandwidth()
        return np.diag(bandwidth_array)

    @staticmethod
    def _parallel_kde_bandwidth(data_package):
        data_slice, cross_validation = data_package
        n_samples = len(data_slice)

        # Not enough data to perform CV
        if n_samples < 2:
            return 0.05

        cross_validation = min(cross_validation, n_samples)
        if cross_validation < 2:
            cross_validation = 2

        #0.01 is too high for daily
        # 0.5 is too low for monthly
        bandwidths = np.arange(1, 20+1) / 100  # Search range: 0.01 to 0.50
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=cross_validation) # 5
        grid.fit(data_slice.reshape(-1, 1))
        return grid.best_params_['bandwidth']

    def _diagonal_kde_bandwidth(self):
        precomputed_bandwidth = self.load_bandwidth()
        if precomputed_bandwidth is not None:
            return precomputed_bandwidth
        
        data = self.returns.to_numpy()
        if not self.std and not self.lock:
            with Pool() as pool:
                optimal_bandwidths = pool.map(self._parallel_kde_bandwidth, [(data[:, d], self.cv) for d in range(self.dim)])
        else:
            optimal_bandwidths = np.std(data, axis=0)
        result = np.array(optimal_bandwidths)
        self.save_bandwidth(result)
        return result

    def _isotropic_kde_bandwidth(self):
        data = self.returns.to_numpy()
        bandwidth_range = np.arange(1, 20+1) / 100 # Search range: 0.01 to 0.50
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidth_range}, cv=self.cv)
        optimal_bandwidth = grid.fit(data).best_params_['bandwidth']
        return np.full(self.dim, optimal_bandwidth)
    
    def save_bandwidth(self, bandwidth: np.ndarray):
        if self.loaded or self.lock:
            return
        
        filename = self._get_bandwidth_filename()
        date = self.returns.index[-1]
        columns = self.returns.columns

        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
        else:
            df = pd.DataFrame(columns=settings.global_tickers)

        df.loc[date, columns] = bandwidth
        df.to_csv(filename)

    def load_bandwidth(self):
        filename = self._get_bandwidth_filename()
        date = self.returns.index[-1]
        columns = self.returns.columns
        df = None

        if self.lock is not None:
            valid, values = self._validate_bandwidth(self.bandwidth_df, date, columns)
            if valid:
                self.loaded = True
                return values
            else: 
                return None   

        if self.bandwidth_df is not None:
            df = self.bandwidth_df
        elif os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0, parse_dates=True)

        valid, values = self._validate_bandwidth(df, date, columns)
        if valid:
            self.loaded = True
            return values       
        
        return None
    
    @staticmethod
    def _validate_bandwidth(bandwidth: pd.DataFrame, date: pd.Timestamp, columns: list) -> bool:
        if bandwidth is not None:
            if date in bandwidth.index:
                values = bandwidth.loc[date, columns]
                if values.notna().all():
                    return True, values.to_numpy()
        return False, None

    def _get_bandwidth_filename(self):
        if self.std:
            base_dir = f'data/kde_bandwidths_std/{settings.index_name}'
        else:
            base_dir = f'data/kde_bandwidths/{settings.index_name}'
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.join(
            base_dir,
            f"{settings.data_frequency}_{settings.window_size}_{settings.window_unit}.csv"
        )
        return filename
        
class GMM():
    def __init__(self, returns, k: int=3, k_min: int=1, k_max: int=10, method: str='default', cv: int=5, random_state: int=19, lock: bool = None, **kwargs):
        self.returns = returns
        self.k = k
        self.k_min = k_min
        self.k_max = k_max
        self.method = method.lower()
        self.cv = cv
        self.random_state = random_state
        self.lock = lock

        assert self.method in ('default', 'cross_validate', 'aic'), "Invalid method. Valid methods are: 'default', 'cross_validate', 'aic'."
        if self.lock is not None:
            assert self.method == 'default', "Lock can only be used with the 'default' method."
        self.gmm_moments = self._fit_gmm_moments()

    def _fit_gmm_moments(self) -> tuple:
       data = self.returns.to_numpy()

       methods = {
            'default': self._fit_gmm_parametric,
            'cross_validate': self._cross_validate_gmm, 
            'aic': self._select_gmm_via_aic}
       
       arguments = {
            'default': (data, self.k, False, self.random_state),
            'cross_validate': (data, self.k_min, self.k_max, self.cv), 
            'aic': (data, self.k_min, self.k_max)}
       
       weights, means, covariances = methods[self.method](*arguments[self.method])
    
       return (weights, means*settings.ANNUALIZATION_FACTOR, covariances*settings.ANNUALIZATION_FACTOR)
       return methods[self.method](*arguments[self.method])

    @staticmethod
    def _fit_gmm_parametric(data: np.ndarray, n_components: int = 3, internal: bool = False, random_state: int = 19) -> tuple:
        n_samples = len(data)
        if n_samples < n_components:
            n_components = max(2, n_samples)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state).fit(data)
        return gmm if internal else (gmm.weights_, gmm.means_, gmm.covariances_)
    
    def _select_gmm_via_aic(self, data: np.ndarray, k_min: int = 1, k_max: int = 10) -> tuple:
        """Finds the optimal number of components (K) for a GMM using AIC."""
        ic_scores = {'aic': [], 'bic': []}
        best_k_bic = None
        best_k_aic = None
        # best_gmm_bic = None
        # best_gmm_aic = None
        best_aic = np.inf
        best_bic = np.inf

        k_args = ((data, k, self.random_state) for k in range(k_min, k_max + 1))

        def evaluate_gmm(args):
            data, k, _random_state = args
            # print(k)
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=_random_state, n_init=10).fit(data)
            return gmm.aic(data), gmm.bic(data), gmm.n_components, gmm

        with Pool() as pool:
            scores = pool.map(evaluate_gmm, k_args)

        for aic, bic, k, gmm in scores:
            ic_scores['aic'].append((aic, k))
            ic_scores['bic'].append((bic, k))

            if aic < best_aic:
                best_aic = aic
                best_k_aic = k
                # best_gmm_aic = gmm
            if bic < best_bic:
                best_bic = bic
                best_k_bic = k
                # best_gmm_bic = gmm

        tqdm.write(f"Best K (AIC): {best_k_aic}")
        tqdm.write(f"Best AIC: {best_aic}")
        tqdm.write(f"Best K (BIC): {best_k_bic}")
        tqdm.write(f"Best BIC: {best_bic}")
        self.aic_curve = sorted([(k, score) for score, k in ic_scores['aic']], key=lambda x: x[0])
        self.bic_curve = sorted([(k, score) for score, k in ic_scores['bic']], key=lambda x: x[0])

        
    def _cross_validate_gmm(self, data: np.ndarray, k_min: int = 1, k_max: int = 10, n_splits: int = 5) -> tuple:
        """Performs cross-validation to find the optimal number of GMM components (K)."""
        _random_state = self.random_state
        best_k, best_gmm, best_cv_score = None, None, -np.inf
        cv_scores = []
        data = self.returns.to_numpy()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=_random_state)

        def evaluate_fold(args):
            train_idx, val_idx, k = args
            gmm = self._fit_gmm_parametric(data[train_idx], k, internal=True, random_state=_random_state)
            return k, gmm.score(data[val_idx])

        fold_jobs = [(train_idx, val_idx, k) for k in range(k_min, k_max + 1) for train_idx, val_idx in kf.split(data)]
        with Pool() as pool:
            results = pool.map(evaluate_fold, fold_jobs)

        k_log_likelihoods = {k: [] for k in range(k_min, k_max + 1)}
        for k, score in results:
            k_log_likelihoods[k].append(score)

        for k, scores in k_log_likelihoods.items():
            mean_cv_score = np.mean(scores)
            cv_scores.append((k, mean_cv_score))

            if mean_cv_score > best_cv_score:
                best_cv_score = mean_cv_score
                best_k = k

        best_gmm = self._fit_gmm_parametric(data, best_k, internal=True, random_state=_random_state)
        print(f"Best K (CV): {best_k}")
        self.cv_curve = sorted([(k, score) for k, score in cv_scores], key=lambda x: x[0])
        return best_gmm.weights_, best_gmm.means_, best_gmm.covariances_
    
class Moments():
    def __init__(self, returns: pd.DataFrame | pd.Series, eval_returns: pd.DataFrame | pd.Series, risk_free_rate: pd.Series | float = 0, market_values: pd.Series = None):
        self.returns = returns
        self.returns_np = returns.values
        self.ticker = self.returns.columns
        self.rf = risk_free_rate
        self.expected_returns = self.get_expected_returns()
        self.expected_covariance = self.get_expected_covariance()
        self.dim = len(self.expected_returns)
        self.len = len(self.returns)

        self.eval_returns = eval_returns
        self.eval_index = self.eval_returns.index
        self.market_values = market_values[self.ticker] if market_values is not None else None
        self.type = 'moments_only'
        self.returns_np = None

    def get_expected_returns(self) -> pd.Series:
        """Optimized expected returns calculation using NumPy."""
        # Convert to numpy, calculate, then convert back to pandas
        mean_np = np.nanmean(self.returns_np, axis=0)
        mean_np = np.where(np.isfinite(mean_np), mean_np, -1.0)
        return pd.Series(mean_np * settings.ANNUALIZATION_FACTOR, index=self.ticker)
        
    def get_expected_covariance(self) -> pd.DataFrame:
        """Optimized covariance calculation using NumPy."""
        
        # Check for null variance assets using NumPy
        var_np = np.nanvar(self.returns_np, axis=0)
        null_indices = np.where(var_np == 0)[0]
        
        # Calculate covariance directly with NumPy
        varcov_np = np.cov(self.returns_np, rowvar=False, ddof=0)
        
        # Handle null variance assets
        if len(null_indices) > 0:
            varcov_np[null_indices, :] = 0
            varcov_np[:, null_indices] = 0
            for idx in null_indices:
                varcov_np[idx, idx] = 100 + 10 * np.random.rand()
        
        # Convert back to pandas
        varcov_matrix = pd.DataFrame(
            varcov_np * settings.ANNUALIZATION_FACTOR, 
            index=self.ticker, 
            columns=self.ticker
        )
        
        return varcov_matrix

class Portfolio():
    valid_types = ('erc', 'max_sharpe', 'min_var', 'equal', 'kde', 'gmm', 'kde_max_sharpe', 'gmm_max_sharpe', 'moments_only', 'value', 'markowitz')

    kde_defaults = {'h': None, 'gamma': 1, 'matrix': 'isotropic', 'cv': 5}
    gmm_defaults = {'k': 3, 'gamma': 1, 'method': 'default', 'k_min': 1, 'k_max': 10, 'cv': 5, 'solver': 'ipopt'}
    markowitz_defaults = {'gamma': 1}

    valid_solvers_class = {
        'erc': ['cvxpy', 'scipy', 'ipopt'], 
        'equal': ['cvxpy', 'scipy', 'ipopt'],
        'value': ['cvxpy', 'scipy', 'ipopt'],
        'max_sharpe': ['cvxpy', 'scipy', 'ipopt'], 
        'min_var': ['cvxpy', 'scipy'], 
        'kde': ['cvxpy', 'ipopt', 'scipy'],
        'gmm': ['ipopt', 'cvxpy', 'scipy'],
        'kde_max_sharpe': ['cvxpy', 'scipy', 'ipopt'],
        'gmm_max_sharpe': ['cvxpy', 'scipy', 'ipopt'],
        'markowitz': ['cvxpy', 'scipy'],
    }
    
    non_combined_portfolios = []

    def __init__(self, 
                 returns: pd.DataFrame | pd.Series = None, 
                 type: str = 'min_var', 
                 risk_free_rate: float = 0, 
                 trust_markowitz: bool = False, 
                 resample: bool = False, 
                 main: bool = False,
                 erc_child: bool = False,
                 solver: str = None, 
                 robust: bool = False,
                 prefit_kde: KDE = None,
                 prefit_gmm: GMM = None,
                 market_values: pd.Series = None,
                 moments_instance: "Portfolio" = None,
                 verbose: bool = False, 
                 args: dict = {}):
        
        self.verbose = verbose
        self.type = type.lower()
        self.valid_solvers = deepcopy(self.valid_solvers_class)
        self.resample = resample
        self.trust_markowitz = trust_markowitz
        self.solver = solver.lower() if solver else None
        self.robust = robust
        self.erc_child = erc_child

        if self._use_moments_instance(returns, risk_free_rate, moments_instance, market_values):
            return  
        self._validate_inputs(args, prefit_kde, prefit_gmm, main)

        # if self.dim <= 10 and self.type in ('max_sharpe', 'kde_max_sharpe', 'gmm_max_sharpe'):
        #     self.solver = 'scipy'

        self._initialize_kde_gmm(args, prefit_kde, prefit_gmm)

        self.optimal_weights = self.get_optimize()
        self.expected_portfolio_return = self.get_expected_portfolio_return()
        self.expected_portfolio_varcov = self.get_expected_portfolio_varcov()

        if (self.type != 'erc' or not main) and self.erc_child:
            Portfolio.non_combined_portfolios.append(self)
        

    def _validate_inputs(self, args, prefit_kde, prefit_gmm, main):
        assert self.type in self.valid_types, f"Invalid type: {self.type}. Valid types are: {self.valid_types}"
        # if self.type in ('kde', 'gmm') and not prefit_kde:
        #     assert len(args) != 0, "KDE and GMM require additional arguments."
        assert main or not self.trust_markowitz, "Non-main portfolios cannot trust Markowitz."
        if self.solver:
            assert self.solver in self.valid_solvers_class[self.type], f"Invalid solver: {self.solver}. Valid solvers: {self.valid_solvers_class[self.type]}"
        if self.type == 'erc' and self.returns.isna().all().all() and not self.trust_markowitz:
            if self.verbose:
                tqdm.write("ERC sample is empty. Falling back to ex-ante expectations.")
        
            
            self.trust_markowitz = True
        if self.type == 'value' and self.market_values is None:
            raise ValueError("Market values are required for 'value' type.")

    def _use_moments_instance(self, returns, rf_rate, moments_instance, market_values):
        if moments_instance and not (self.trust_markowitz and self.type == 'erc'):
            assert moments_instance.type != "erc", "Using 'erc' as a moments instance can result in unexpected behavior."
            self.returns = moments_instance.returns
            self.ticker = moments_instance.ticker
            self.rf = moments_instance.rf
            self.expected_returns = moments_instance.expected_returns
            self.expected_covariance = moments_instance.expected_covariance
            self.dim = moments_instance.dim
            self.len = moments_instance.len
            self.market_values = moments_instance.market_values if self.type == 'value' else None
        
        else:
            self.returns = returns
            self.ticker = returns.columns
            self.rf = rf_rate
            self.expected_returns = self.get_expected_returns()
            self.expected_covariance = self.get_expected_covariance()
            self.dim = len(self.expected_returns)
            self.len = len(self.returns)
            self.market_values = market_values if self.type == 'value' else None

        return self.type == 'moments_only'

    def _initialize_kde_gmm(self, args, prefit_kde, prefit_gmm):
        if self.type in ('kde', 'kde_max_sharpe'):
            self.kde_args = {key: args.get(key, item) for key, item in self.kde_defaults.items()}
            self.bandwidth_matrix = prefit_kde.bandwidth_matrix if prefit_kde else KDE(self.returns, **self.kde_args).bandwidth_matrix
            self.returns *= settings.ANNUALIZATION_FACTOR

        if self.type in ('gmm', 'gmm_max_sharpe'):
            self.gmm_args = {key: args.get(key, item) for key, item in self.gmm_defaults.items()}
            self.gmm_moments = prefit_gmm.gmm_moments if prefit_gmm else GMM(self.returns, **self.gmm_args).gmm_moments

        if self.type == 'markowitz':
            self.markowitz_args = {key: args.get(key, item) for key, item in self.markowitz_defaults.items()}


    def get_expected_returns(self) -> pd.DataFrame | pd.Series:
        #TODO: Attention! If extending beyond ERC, if statement must be updated.
        if self.type == 'erc':
            if self.trust_markowitz:
                internal_expectations = np.array([portfolio.expected_portfolio_return for portfolio in Portfolio.non_combined_portfolios])
                return pd.Series(internal_expectations, index=self.returns.columns)
            elif self.returns.eq(0).all().all():
                return self.returns.mean(axis=0)
        return self.returns.mean(axis=0) * settings.ANNUALIZATION_FACTOR
    
    def get_expected_covariance(self) -> pd.DataFrame | pd.Series:
        if self.trust_markowitz and self.type == 'erc':
            internal_expectations = np.array([np.sqrt(portfolio.expected_portfolio_varcov) for portfolio in Portfolio.non_combined_portfolios])
            sample_correlations = self.returns.corr().fillna(0)
            varcov_matrix = np.outer(internal_expectations, internal_expectations) * sample_correlations
            varcov_matrix = pd.DataFrame(varcov_matrix, index=self.returns.columns, columns=self.returns.columns)
        else:
            varcov_matrix = self.returns.cov(ddof=0)

        # Poison the covariance matrix for invalid assets
        null_variance_assets = self.returns.var(axis=0).eq(0)
        null_variance_assets = null_variance_assets[null_variance_assets].index.tolist()
        varcov_matrix.loc[null_variance_assets, :] = 0
        varcov_matrix.loc[:, null_variance_assets] = 0
        varcov_matrix.loc[null_variance_assets, null_variance_assets] = 100 + 10 * np.random.rand()

        return varcov_matrix * settings.ANNUALIZATION_FACTOR
    
    # TODO: Check annualization factor
    def get_expected_portfolio_return(self) -> float:
        return np.dot(self.expected_returns, self.optimal_weights)
        return np.dot(self.expected_returns, self.optimal_weights) * settings.ANNUALIZATION_FACTOR
    
    def get_expected_portfolio_varcov(self) -> float:
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights
        return self.optimal_weights.T @ self.expected_covariance @ self.optimal_weights * settings.ANNUALIZATION_FACTOR ** 2

    def _select_method(self, failed: str=None):
        #TODO: Separate min_var and markowitz
        if self.type == 'equal':
            return self._fit_equal
        if self.type == 'value':
            return self._fit_value
        
        if self.solver:
            solver = self.solver
        if not self.solver and not failed:
            solver = self.valid_solvers[self.type][0]
        if failed:
            if len(self.valid_solvers[self.type]) == 1:
                raise RuntimeError(f"All solvers failed for {self.type}.")
            else:
                self.valid_solvers[self.type].remove(failed)
                solver = self.valid_solvers[self.type][0]
                self.solver = solver
                if self.verbose:
                    tqdm.write(f"{self.type.upper()}: Solver {failed} failed. Trying {solver}.")
                
        solver_map = {
                ('erc', 'cvxpy'): self._fit_erc_CVXPY,
                ('erc', 'scipy'): self._fit_erc_SCIPY,
                ('erc', 'ipopt'): self._fit_erc_IPOPT,
                ('max_sharpe', 'cvxpy'): self._fit_max_sharpe_CVXPY,
                ('max_sharpe', 'scipy'): self._fit_max_sharpe_SCIPY,
                ('max_sharpe', 'ipopt'): self._fit_max_sharpe_IPOPT,
                ('min_var', 'cvxpy'): self._fit_min_var_CVXPY,
                ('min_var', 'scipy'): self._fit_min_var_SCIPY,
                ('kde', 'cvxpy'): self._fit_KDE_CVXPY,
                ('kde', 'scipy'): self._fit_KDE_SCIPY,
                ('kde', 'ipopt'): self._fit_KDE_IPOPT,
                ('gmm', 'ipopt'): self._fit_GMM_IPOPT,
                ('gmm', 'cvxpy'): self._fit_GMM_CVXPY,
                ('gmm', 'scipy'): self._fit_GMM_SCIPY,
                ('kde_max_sharpe', ...): self._fit_KDE_max_sharpe,
                ('gmm_max_sharpe', ...): self._fit_GMM_max_sharpe,
                ('markowitz', 'cvxpy'): self._fit_markowitz_CVXPY,
                ('markowitz', 'scipy'): self._fit_markowitz_SCIPY,
            }

        if (self.type, solver) in solver_map:
            try:
                return solver_map[(self.type, solver)]
            except Exception as _:
                if self.verbose:
                    tqdm.write(f"Routine {self.type.upper()} with solver {solver} failed.")
                return self._select_method(failed=solver)
        elif (self.type, ...) in solver_map:
            return solver_map[(self.type, ...)]
        else:
            raise RuntimeError(f"No valid fitting method found for type '{self.type}' and solver '{solver}'.")
    
    def _fit_GMM_CVXPY(self) -> np.ndarray:
        gamma = self.gmm_args['gamma']
        gmm_weights, gmm_means, gmm_covariances = self.gmm_moments
        K, d = gmm_means.shape
        log_phi = np.log(gmm_weights)
        weights = cp.Variable(d)
        
        quad_terms = [0.5 * gamma**2 * cp.quad_form(weights, gmm_covariances[i]) for i in range(K)]
        linear_terms = [-gamma * (gmm_means[i] @ weights) for i in range(K)]
        f_terms = [log_phi[i] + linear_terms[i] + quad_terms[i] for i in range(K)]
        lse_objective = cp.log_sum_exp(cp.hstack(f_terms))
        
        objective = cp.Minimize(lse_objective)
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(objective, constraints)

        timeout = settings.timeout  # seconds
        start_time = time.time()

        for solver in ['CLARABEL', 'SCS']:
            try:
                problem.solve(solver=solver, warm_start=True) # Deafult: CLARABEL
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Solver {solver} exceeded {timeout} seconds.")
                if weights.value is not None:
                    # print(f"Objective value: {problem.value}; weight sum: {np.sum(weights.value)}")
                    return weights.value
                
            except Exception as e:
                if self.verbose:
                    if isinstance(e, TimeoutError):
                        tqdm.write(f"Solver {solver} exceeded {timeout} seconds.")
                    else:
                        tqdm.write(f"Solver {solver} failed with error: {e}")

        raise RuntimeError("All CVXPY solvers failed")
    
    def _fit_GMM_IPOPT(self) -> np.ndarray:
        gamma = self.gmm_args['gamma']
        gmm_weights, gmm_means, gmm_covariances = self.gmm_moments
        K, dim = gmm_means.shape
        log_phi = np.log(gmm_weights)
        quad_const = 0.5 * gamma**2
        quad_terms_constant = np.full(K, quad_const)
        
        last_vector = None
        cached = [None, None, None]

        def compute_common_values(w):
            nonlocal cached, last_vector
            hashed_vector = hash(w.tobytes())
            if last_vector is not None and last_vector == hashed_vector:
                return cached

            quad_terms = quad_terms_constant * np.sum(w * (gmm_covariances @ w[:, None]).squeeze(-1), axis=1)
            linear_terms = -gamma * np.sum(w * gmm_means, axis=1)
            softmax_weights = softmax(log_phi + linear_terms + quad_terms)

            last_vector = hashed_vector
            cached = [quad_terms, linear_terms, softmax_weights]
            return quad_terms, linear_terms, softmax_weights

        def objective(w):
            quad_terms, linear_terms, _ = compute_common_values(w)
            return logsumexp(log_phi + linear_terms + quad_terms)
        
        def jacobian(w):
            _, _, softmax_weights = compute_common_values(w)
            return np.sum(softmax_weights[:, None] * (-gamma * gmm_means + gamma**2 * np.dot(gmm_covariances, w)), axis=0)
        
        def hessian(w):
            _, _, softmax_weights = compute_common_values(w)

            H = np.zeros((len(w), len(w)))
            Z = [gamma**2 * (gmm_covariances[i] @ w) - gamma * gmm_means[i] for i in range(K)]  # ∇z_i(w)
            sum_pZ = np.zeros_like(Z[0])
            
            for i in range(K):
                p = softmax_weights[i]
                Z_i = Z[i]
                C_i = gmm_covariances[i]
                H += p * (gamma**2 * C_i + np.outer(Z_i, Z_i))
                sum_pZ += p * Z_i

            H -= np.outer(sum_pZ, sum_pZ)
            return H
        
        def constraint_eq(w):
            return np.sum(w) - 1  

        def constraint_eq_jac(_):
            return np.ones((1, dim)) 

        def constraint_eq_hess(*_):
            return np.zeros((dim, dim))

        kwargs = {
            'fun': objective,
            'jac': jacobian,
            'hess': hessian,
            'x0': np.ones(dim) / dim,
            'bounds': [(0, 1) for _ in range(dim)],
            'constraints': {'type': 'eq', 
                            'fun': constraint_eq,
                            'jac': constraint_eq_jac,
                            'hess': constraint_eq_hess},
            'options': {'tol': 1e-12,
                        'constr_viol_tol': 1e-10,
                        'max_iter': 2000,
                        # 'max_cpu_time': 10.0,
                        'honor_original_bounds': 'yes',
                        'sb': 'yes'
                        }}

        return minimize_ipopt(**kwargs).x

    def _fit_GMM_SCIPY(self) -> np.ndarray:
        gamma = self.gmm_args['gamma']
        gmm_weights, gmm_means, gmm_covariances = self.gmm_moments
        K, dim = gmm_means.shape
        log_phi = np.log(gmm_weights)
        quad_terms_constant = np.full(K, 0.5 * gamma**2)
        
        def objective(w):
            quad_terms = quad_terms_constant * np.sum(w * (gmm_covariances @ w[:, None]).squeeze(-1), axis=1)
            linear_terms = -gamma * np.sum(w * gmm_means, axis=1)
            return logsumexp(log_phi + linear_terms + quad_terms)
        
        def jacobian(w):
            quad_terms = quad_terms_constant * np.sum(w * (gmm_covariances @ w[:, None]).squeeze(-1), axis=1)
            linear_terms = -gamma * np.sum(w * gmm_means, axis=1)
            softmax_weights = softmax(log_phi + linear_terms + quad_terms) 
            return np.sum(softmax_weights[:, None] * (-gamma * gmm_means + gamma**2 * np.dot(gmm_covariances, w)), axis=0)

        kwargs = {
            'fun': objective,
            'jac': jacobian,
            'x0': np.ones(dim) / dim,
            'constraints': LinearConstraint(np.ones((1, dim)), 1, 1),
            'bounds': Bounds(0, 1),
            'method': 'SLSQP',
            'tol': 1e-12}
        
        return minimize(**kwargs).x  

    def _fit_GMM_max_sharpe(self) -> np.ndarray:
        gmm_weights, gmm_means, gmm_covariances = self.gmm_moments
        weighted_means = np.sum(gmm_means * gmm_weights[:, None], axis=0)
        weighted_covariances = np.sum([gmm_weights[i] * gmm_covariances[i] for i in range(len(gmm_weights))], axis=0)
        central_means = gmm_means - weighted_means
        total_variance = weighted_covariances + np.sum([gmm_weights[i] * np.outer(central_means[i], central_means[i]) for i in range(len(gmm_weights))], axis=0)

        # weighted_means = self._pandify(weighted_means) * settings.ANNUALIZATION_FACTOR
        # total_variance = self._pandify(total_variance) * settings.ANNUALIZATION_FACTOR
        weighted_means = self._pandify(weighted_means)
        total_variance = self._pandify(total_variance)

        backup_means, backup_covariances = self.expected_returns.copy(), self.expected_covariance.copy()
        self.expected_returns, self.expected_covariance = weighted_means, total_variance
        self.type = 'max_sharpe'

        result = self._select_method()()

        self.expected_returns, self.expected_covariance = backup_means, backup_covariances
        self.type = 'gmm_max_sharpe'
        return result

    def _fit_KDE_CVXPY(self) -> np.ndarray:

        bandwidth_matrix = self.bandwidth_matrix
        gamma = self.kde_args['gamma']
        weights = cp.Variable(self.dim)
        np_returns = self.returns.to_numpy()

        objective = cp.Minimize(0.5 * gamma**2 * cp.quad_form(weights, bandwidth_matrix) + cp.log_sum_exp(-gamma * np_returns @ weights))
        constraints = [cp.sum(weights) == 1, 
                       weights >= 0]
        problem = cp.Problem(objective, constraints)

        for solver in ['CLARABEL', 'SCS']:
            try:
                problem.solve(solver=solver, warm_start=True) # Deafult: Clarabel
                if weights.value is not None:
                    return weights.value
            except Exception as _:
                if self.verbose:
                    tqdm.write(f"KDE_CVXPY: Solver {solver} failed.")

        raise RuntimeError("All CVXPY solvers failed")
    
    def _fit_KDE_IPOPT(self) -> np.ndarray:
        bandwidth_matrix = self.bandwidth_matrix
        gamma = self.kde_args['gamma']
        np_returns = self.returns.to_numpy()
        N = self.len
        dim = self.dim

        def objective(w):
            return 0.5 * gamma**2 * np.dot(w, np.dot(bandwidth_matrix, w)) + logsumexp(-gamma * np.dot(np_returns, w))
        
        def jacobian(w):
            return gamma**2 * np.dot(bandwidth_matrix, w) - gamma * np.sum(softmax(-gamma * np.dot(np_returns, w))[:, None] * np_returns, axis=0)
        
        def hessian(w):
            # H = gamma**2 * bandwidth_matrix.copy()
            H = gamma**2 * bandwidth_matrix
            weights = softmax(-gamma * np.dot(np_returns, w))               
            sum_pR = np.sum(weights[:, None] * np_returns, axis=0)

            RR = np.zeros_like(H)
            for i in range(N):
                ri = np_returns[i]        
                RR += weights[i] * np.outer(ri, ri)
            H += gamma**2 * (RR - np.outer(sum_pR, sum_pR))
            return H
            
        def constraint_eq(w):
            return np.sum(w) - 1  

        def constraint_eq_jac(_):
            return np.ones((1, dim)) 

        def constraint_eq_hess(*_):
            return np.zeros((dim, dim))

        kwargs = {
            'fun': objective,
            'jac': jacobian,
            'hess': hessian,
            'x0': np.ones(dim) / dim,
            'bounds': [(0, 1) for _ in range(dim)],
            'constraints': {'type': 'eq', 
                            'fun': constraint_eq,
                            'jac': constraint_eq_jac,
                            'hess': constraint_eq_hess},
            'options': {'tol': 1e-12,
                        'constr_viol_tol': 1e-10,
                        'max_iter': 2000,
                        # 'max_cpu_time': 10.0,
                        'honor_original_bounds': 'yes',
                        'sb': 'yes'
                        }}

        return minimize_ipopt(**kwargs).x

    def _fit_KDE_SCIPY(self) -> np.ndarray:
        bandwidth_matrix = self.bandwidth_matrix
        gamma = self.kde_args['gamma']

        kwargs = {
            'fun': lambda x: 0.5 * gamma**2 * np.dot(x, np.dot(bandwidth_matrix, x)) + np.log(np.sum(np.exp(-gamma * np.dot(self.returns, x)))),
            'jac': lambda x: gamma**2 * np.dot(bandwidth_matrix, x) - gamma * np.sum(softmax(-gamma * np.dot(self.returns, x))[:, None] * self.returns, axis=0),
            'x0': np.ones(self.dim) / self.dim,
            'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
            'bounds': Bounds(0, 1),
            'method': 'SLSQP',
            'tol': 1e-6} #'tol': 1e-16
    
        return minimize(**kwargs).x
    
    def _fit_KDE_max_sharpe(self) -> np.ndarray:
        self.expected_covariance += self.bandwidth_matrix
        self.type = 'max_sharpe'
        result = self._select_method()()
        self.type = 'kde_max_sharpe'
        self.expected_covariance -= self.bandwidth_matrix
        return result

    def _fit_equal(self) -> np.ndarray:
        return np.ones(self.dim) / self.dim
    
    def _fit_value(self) -> np.ndarray:
        # print((self.market_values / np.sum(self.market_values)).values)
        return (self.market_values / np.sum(self.market_values)).values
            
    def get_optimize(self) -> np.ndarray:
        "Returns a numpy array of optimal weights"
        if self.resample:
            return self._resample()
        else:
            try:
                return self._select_method()()
            except Exception as _:
                return self._select_method(failed=self.valid_solvers[self.type][0])()

    def _resample(self) -> np.ndarray:
        if self.type in ('kde', 'gmm'):
            raise NotImplementedError("Resampling is not supported for KDE and GMM.")
        
        N_SUMULATIONS = 10 # 500

        method = self._select_method()
        original_moments = (self.expected_returns.copy(), self.expected_covariance.copy())
        simulated_weights = []

        for i in range(N_SUMULATIONS):
            np.random.seed(i)
            simulated_returns = np.random.multivariate_normal(self.expected_returns, self.expected_covariance, self.len)
            # TODO: verify necessity of annualization factor
            self.expected_returns = self._pandify(np.mean(simulated_returns, axis=0))
            self.expected_covariance = self._pandify(np.cov(simulated_returns.T, ddof=0))# * settings.ANNUALIZATION_FACTOR
            simulated_weights.append(method())
        
        self.expected_returns, self.expected_covariance = original_moments
        combined_simulation_data = np.stack(simulated_weights, axis=0)
        # return pd.Series(combined_simulation_data.mean(axis=0), index=self.ticker)
        return combined_simulation_data.mean(axis=0)
    
    def _pandify(self, array: np.ndarray) -> pd.Series | pd.DataFrame:
        if array.ndim == 1:
            return pd.Series(array, index=self.ticker)
        else:
            return pd.DataFrame(array, index=self.ticker, columns=self.ticker)
           
    def _fit_min_var_CVXPY(self) -> np.ndarray:   
        weights = cp.Variable(self.dim)
        portfolio_variance = cp.quad_form(weights, cp.psd_wrap(self.expected_covariance))
        objective = cp.Minimize(portfolio_variance)
        constraints = [cp.sum(weights) == 1, 
                       weights >= 0]
        problem = cp.Problem(objective, constraints)

        for solver in ['CLARABEL', 'SCS']:
            try:
                problem.solve(solver=solver, warm_start=True) # Deafult: Clarabel
                if weights.value is not None:
                    return weights.value
            except Exception as _:
                if self.verbose:
                    tqdm.write(f"MIN_VAR_CVXPY: Solver {solver} failed.")

        raise RuntimeError("All CVXPY solvers failed")
    
    def _fit_min_var_SCIPY(self) -> np.ndarray:
        Sigma = self.expected_covariance
        kwargs = {'fun': lambda x: np.dot(x, np.dot(Sigma, x)),
                'jac': lambda x: 2 * np.dot(Sigma, x),
                'x0': np.ones(self.dim) / self.dim,
                'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
                'bounds': Bounds(0, 1),
                'method': 'SLSQP',
                'tol': 1e-10} #'tol': 1e-16
        return minimize(**kwargs).x
    
    def _fit_markowitz_SCIPY(self):
        initial_guess = np.ones(self.dim) / self.dim 
        constraints = [LinearConstraint(np.ones(self.dim), 1, 1)]
        bounds = Bounds(0, 1)
        gamma = self.markowitz_args['gamma']

        def objective(weights):
            return 0.5 * np.dot(weights.T, np.dot(self.expected_covariance, weights)) - gamma * np.dot(self.expected_returns, weights)
        def jacobian(weights):
            return np.dot(self.expected_covariance, weights) - gamma * self.expected_returns

        kwargs = {'fun': objective,
                'jac': jacobian,
                'x0': initial_guess,
                'constraints': constraints,
                'bounds': bounds,
                'method': 'SLSQP',
                'tol': 1e-16}
        return minimize(**kwargs).x

    def _fit_markowitz_CVXPY(self):
        gamma = self.markowitz_args['gamma']
        weights = cp.Variable(self.dim)
        markowitz = 0.5 * cp.quad_form(weights, cp.psd_wrap(self.expected_covariance)) - gamma * weights @ self.expected_returns
        constraints = [cp.sum(weights) == 1, weights >= 0]
        problem = cp.Problem(cp.Minimize(markowitz), constraints)

        for solver in ['CLARABEL', 'SCS']:
            try:
                problem.solve(solver=solver, warm_start=True) # Deafult: Clarabel
                if weights.value is not None:
                    return weights.value
            except Exception as _:
                if self.verbose:
                    tqdm.write(f"MARKOWITZ_CVXPY: Solver {solver} failed.")

        raise RuntimeError("All CVXPY solvers failed")
    
    def _fit_max_sharpe_CVXPY(self) -> np.ndarray:
        if self.expected_returns.isna().all().all() or (self.expected_returns == 0).all().all():
            return np.zeros(self.dim)
        
        proxy_weights = cp.Variable(self.dim)
        # objective = cp.Minimize(cp.quad_form(proxy_weights, self.expected_covariance))
        objective = cp.Minimize(cp.quad_form(proxy_weights, cp.psd_wrap(self.expected_covariance)))
        constraints = [proxy_weights @ (self.expected_returns - self.rf) == 1, 
                    proxy_weights >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(warm_start=True)

        # print(proxy_weights.value)
        if proxy_weights.value is None:
            raise RuntimeError("All CVXPY solvers failed")
            result = self._fit_max_sharpe_robust()
        else:
            result = proxy_weights.value / np.sum(proxy_weights.value)
        return result
    
    def _fit_max_sharpe_IPOPT(self):
        if self.verbose:
            tqdm.write("Warning: IPOPT is not recommended for Max Sharpe optimization.")
        mu = self.expected_returns - self.rf
        Sigma = self.expected_covariance
        dim = self.dim

        def objective(w):
            return np.dot(w, Sigma @ w)

        def jacobian(w):
            return 2.0 * (Sigma @ w)

        def hessian(*_):
            return 2.0 * Sigma
        
        def constraint_eq(w):
            return np.dot(w, mu) - 1

        def constraint_eq_jac(_):
            return mu.values[None, :]

        def constraint_eq_hess(*_):
            return np.zeros((dim, dim))

        mu_sum = np.sum(mu)
        w0 = np.ones(dim) / dim if mu_sum <= 0 else (1/mu_sum) * np.ones(dim)

        kwargs = {
            'fun': objective,
            'jac': jacobian,
            'hess': hessian,
            'x0': w0,
            'bounds': [(0, None) for _ in range(dim)],
            'constraints': {'type': 'eq', 
                            'fun': constraint_eq,
                            'jac': constraint_eq_jac,
                            'hess': constraint_eq_hess},
            'options': {'tol': 1e-12,
                        'constr_viol_tol': 1e-10,
                        'max_iter': 2000,
                        # 'max_cpu_time': 10.0,
                        'honor_original_bounds': 'yes',
                        'sb': 'yes'
                        }}

        proxy_weights = minimize_ipopt(**kwargs).x
        return proxy_weights / np.sum(proxy_weights)
    
    def _fit_max_sharpe_SCIPY(self) -> np.ndarray:
        try:
            mu = self.expected_returns - self.rf
            Sigma = self.expected_covariance
            n = self.dim

            c_sigma = np.ascontiguousarray(Sigma.to_numpy(), dtype=np.float64)
            c_mu = np.ascontiguousarray(mu.to_numpy(), dtype=np.float64)
            grad_out = np.ascontiguousarray(np.empty_like(c_mu), dtype=np.float64)
            ptr_sigma = c_sigma.ctypes.data_as(POINTER(c_double))
            ptr_mu = c_mu.ctypes.data_as(POINTER(c_double))
            ptr_grad_out = grad_out.ctypes.data_as(POINTER(c_double))
            size_t = c_size_t(n)
            
            def objective_and_jac(w):
                ptr_w = w.ctypes.data_as(POINTER(c_double))
                val = lib.max_sharpe_objective_and_jacobian(ptr_w, ptr_sigma, ptr_mu, ptr_grad_out, size_t)
                return val, grad_out

            kwargs = {'fun': objective_and_jac,
                    'jac': True,
                    'x0': np.ones(self.dim) / self.dim,
                    'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
                    'bounds': Bounds(0, 1),
                    'method': 'SLSQP',
                    'tol': 1e-9} #'tol': 1e-16
            return minimize(**kwargs).x
        
        # C mode unavalible, fallback to Python implementation
        except Exception as _:
            tqdm.write("C mode for Max Sharpe is not available, falling back to Python implementation.")
            mu = self.expected_returns - self.rf
            Sigma = self.expected_covariance

            def objective_and_jac(w):
                denominator = np.linalg.multi_dot([w, Sigma, w])
                if denominator <= 1e-12: 
                    return 1e6 
                objective = -np.dot(mu, w) / np.sqrt(denominator)
                jacobian = -(mu / np.sqrt(denominator) - np.dot(mu, w) * np.dot(w, Sigma) / denominator**1.5)
                return objective, jacobian

            kwargs = {'fun': objective_and_jac,
                    'jac': True,
                    'x0': np.ones(self.dim) / self.dim,
                    'constraints': LinearConstraint(np.ones(self.dim), 1, 1),
                    'bounds': Bounds(0, 1),
                    'method': 'SLSQP',
                    'tol': 1e-9} #'tol': 1e-16
            return minimize(**kwargs).x
    
    def _fit_erc_CVXPY(self):
        weights = cp.Variable(self.dim)
        objective = cp.Minimize(0.5 * cp.quad_form(weights, self.expected_covariance))

        log_constraint_bound = -self.dim * np.log(self.dim) - 2  # -2 does not matter after rescaling
        log_constraint = cp.sum(cp.log(weights)) >= log_constraint_bound
        constraints = [weights >= 0, weights <= 1, log_constraint]

        problem = cp.Problem(objective, constraints)
        # problem.solve(solver=cp.SCS, eps=1e-12) # Results in e-27 precision   
        problem.solve(warm_start=True) # Results in e-27 precision    

        if weights.value is None:
            raise RuntimeError("All CVXPY solvers failed")
            result = self._fit_erc_robust()
        else:
            result = weights.value / np.sum(weights.value)
        return result

    def _fit_erc_IPOPT(self) -> np.ndarray:
        Sigma = self.expected_covariance.to_numpy()
        dim = self.dim

        def objective(x):
            sigma = np.sqrt(x @ Sigma @ x)
            v = Sigma @ x
            rc = x * v / sigma
            mean_rc = np.mean(rc)
            return np.sum((rc - mean_rc) ** 2)

        def jacobian(x):
            # 1) Precompute everything
            sigma   = np.sqrt(x @ Sigma @ x)
            v       = Sigma @ x
            rc      = (x * v) / sigma        # shape (dim,)
            mean_rc = rc.mean()
            rc_diff = rc - mean_rc
            
            # 2) Build drc in shape (dim, dim)
            sigma_grad = v / sigma
            drc = np.zeros((dim, dim))
            for k in range(dim):
                s_k = v[k]
                dck = (s_k * np.eye(dim)[k] + x[k]*Sigma[k,:]) / sigma \
                    - (x[k]*s_k/(sigma**2)) * sigma_grad
                drc[k,:] = dck

            drc_mean = drc.mean(axis=0)    # shape (dim,)
            drc_diff = drc - drc_mean      # shape (dim, dim)

            grad = 2.0 * (drc_diff.T @ rc_diff)
            return grad
        
        def hessian(x, vi=None):
            sigma = np.sqrt(x @ Sigma @ x)   # scalar
            v = Sigma @ x                   # shape (dim,)
            rc = (x * v) / sigma            # shape (dim,)
            mean_rc = rc.mean()             # scalar
            rc_diff = rc - mean_rc          # shape (dim,)

            drc = np.zeros((dim, dim))
            d2rc = np.zeros((dim, dim, dim))
            sigma_grad = v / sigma          # shape (dim,)
            sigma_hess = ( np.outer(sigma_grad, sigma_grad) - Sigma ) / sigma

            for k in range(dim):
                s_k = v[k]      # (Sigma x)_k
                part1 = (s_k * np.eye(dim)[k] + x[k]*Sigma[k,:]) / sigma
                part2 = (x[k]*s_k/(sigma**2)) * sigma_grad
                dck   = part1 - part2
                drc[k,:] = dck

                term2 = x[k] * np.outer(Sigma[k,:], sigma_grad)
                term3 = s_k  * np.outer(np.eye(dim)[k], sigma_grad)
                term4 = (x[k]*s_k / sigma**2) * sigma_hess

                d2ck = (term2 + term3)/sigma - term4
                d2rc[k,:,:] = d2ck

            drc_mean = drc.mean(axis=0)       # shape (dim,)
            d2rc_mean = d2rc.mean(axis=0)     # shape (dim, dim)
            drc_diff = drc - drc_mean        # shape (dim, dim)
            d2rc_diff = d2rc - d2rc_mean    # shape (dim, dim, dim)
            H = np.zeros((dim, dim))

            for k in range(dim):
                outer_term = np.outer(drc_diff[k,:], drc_diff[k,:])
                hess_term = d2rc_diff[k,:,:]
                H += 2.0 * ( rc_diff[k] * hess_term + outer_term )

            return H

        def constraint_eq(w):
            return np.sum(w) - 1  

        def constraint_eq_jac(_):
            return np.ones((1, dim)) 

        def constraint_eq_hess(*_):
            return np.zeros((dim, dim))

        w0 = np.full(dim, 1.0 / dim)

        kwargs = {
            'fun': objective,
            'jac': jacobian,
            'hess': hessian,
            'x0': w0,
            'bounds': [(0, None) for _ in range(dim)],
            'constraints': {'type': 'eq', 
                            'fun': constraint_eq,
                            'jac': constraint_eq_jac,
                            'hess': constraint_eq_hess},
            'options': {'tol': 1e-12,
                        'constr_viol_tol': 1e-10,
                        'max_iter': 2000,
                        'honor_original_bounds': 'yes',
                        'sb': 'yes'
                        }}
    
        proxy_weights = minimize_ipopt(**kwargs).x
        return proxy_weights / np.sum(proxy_weights)

    def _fit_erc_SCIPY(self) -> np.ndarray:
        Sigma = self.expected_covariance.to_numpy()
        dim = self.dim

        @njit()
        def _sub_objective(x, SIGMA):
            sigma = np.sqrt(x @ SIGMA @ x)
            v = SIGMA @ x
            rc = (x * v) / sigma
            mean_rc = rc.mean()
            return np.sum((rc - mean_rc) ** 2)
        
        @njit()
        def _sub_jacobian(x, SIGMA, dim):
            # 1) Precompute everything
            sigma   = np.sqrt(x @ SIGMA @ x)
            v       = SIGMA @ x
            rc      = (x * v) / sigma        # shape (dim,)
            mean_rc = rc.mean()
            rc_diff = rc - mean_rc
            
            # 2) Build drc in shape (dim, dim)
            sigma_grad = v / sigma
            drc = np.zeros((dim, dim))
            for k in range(dim):
                s_k = v[k]
                dck = (s_k * np.eye(dim)[k] + x[k]*Sigma[k,:]) / sigma \
                    - (x[k]*s_k/(sigma**2)) * sigma_grad
                drc[k,:] = dck

            # drc_mean = drc.mean(axis=0)    # shape (dim,)
            # drc_mean = np.mean(drc, axis=0)    # shape (dim,)
            drc_mean = np.sum(drc, axis=0) / drc.shape[0]
            drc_diff = drc - drc_mean      # shape (dim, dim)

            grad = 2.0 * (drc_diff.T @ rc_diff)
            return grad

        def objective(x):
            return _sub_objective(x, Sigma)
        
        def jacobian(x):
            return _sub_jacobian(x, Sigma, dim)
            
        
        bounds = Bounds(0, 1)
        lc = LinearConstraint(np.ones(self.dim), 1, 1)
        settings = {'tol': 1e-16, 'method': 'SLSQP'} # This tolerance is required to match cvxpy results
        res = minimize(objective, np.full(self.dim, 1/self.dim), jac=jacobian, constraints=[lc], bounds=bounds, **settings)
        # res = minimize(_ERC, np.full(self.dim, 1/self.dim), constraints=[lc], bounds=bounds, **settings)
        return res.x
    
    def evaluate_performance(self, evaluationData: pd.DataFrame | pd.Series) -> pd.Series:
        # Returns Adjusted for Return-Shifted Weights
        if evaluationData.isna().all().all():
            if self.verbose:
                tqdm.write(f"Warning: Evaluation data is empty for period {evaluationData.index[0]}.")
            self.actual_weights = pd.Series(0, index=evaluationData.index)
            return pd.Series(0, index=evaluationData.index)
        portfolioWeights = self.optimal_weights
        subperiodReturns = []
        subperiodWeights = [portfolioWeights]
        for singleSubperiodReturns in evaluationData.values:
            portfolioReturns = subperiodWeights[-1] @ singleSubperiodReturns
            portfolioWeights = subperiodWeights[-1] * (1 + singleSubperiodReturns) / (1 + portfolioReturns)
            subperiodReturns.append(portfolioReturns)
            subperiodWeights.append(portfolioWeights)
        self.actual_returns = pd.Series(subperiodReturns, index=evaluationData.index)
        self.actual_weights = pd.DataFrame(subperiodWeights[:-1], index=evaluationData.index, columns=self.ticker)
        return pd.Series(subperiodReturns, index=evaluationData.index)
        

# Helper methods
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

def split_large_csv(dataframe, base_path, base_filename="file", max_size_mb=50, indexSet=True):
    import math

    temp_file_path = os.path.join(base_path, f"{base_filename}_temp.csv")
    dataframe.to_csv(temp_file_path, index=indexSet)
    total_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
    os.remove(temp_file_path)

    num_chunks = math.ceil(total_size_mb / max_size_mb)
    if num_chunks > 1:
        chunk_size = math.ceil(len(dataframe) / num_chunks)
        for i in range(num_chunks):
            chunk = dataframe.iloc[i * chunk_size : (i + 1) * chunk_size]
            chunk_path = os.path.join(base_path, f"{base_filename}_part{i + 1}.csv")
            chunk.to_csv(chunk_path, index=indexSet)
        # print(f"DataFrame split into {num_chunks} chunks.")
        tqdm.write(f"DataFrame split into {num_chunks} chunks.")
    else:
        file_path = os.path.join(base_path, f"{base_filename}.csv")
        dataframe.to_csv(file_path, index=indexSet)
        tqdm.write(f"DataFrame saved as a single file: {file_path}")
        # print("DataFrame saved as a single file.")

def iteration_depth(window: int = 5, window_units: str = 'years') -> dict:
    """
    Generate index slices for rolling portfolio evaluation.

    Parameters:
    - window (int): Lookback window in years (for optimization).

    Returns:
    - dict of {step_index: {'optimizationIndex': ..., 'evaluationIndex': ...}}
    """
    master_index = settings.master_index
    frequency = settings.rebalancing_frequency.lower()
    limit_year = settings.limit_year or master_index[-1].year

    if frequency == "annual":
        eval_dates = pd.date_range(start="2006-01-01", end=f"{limit_year}-12-31", freq="YS")
    elif frequency == "monthly":
        eval_dates = pd.date_range(start="2006-01-01", end=f"{limit_year}-12-31", freq="MS")
    else:
        raise ValueError(f"Unsupported rebalancing frequency: {frequency}")
    
    if window_units == 'years':
        window_offset = pd.DateOffset(years=window)
    elif window_units == 'months':
        window_offset = pd.DateOffset(months=window)
    elif window_units == 'days':
        window_offset = pd.DateOffset(days=window)

    index_iterator = {}

    for i, eval_date in enumerate(eval_dates):
        start_date = eval_date - window_offset
        optimization_index = (master_index >= start_date) & (master_index < eval_date)

        if frequency == "monthly":
            eval_end = eval_date + pd.offsets.MonthEnd(1)
        else:
            eval_end = eval_date + pd.offsets.YearEnd(1)  

        evaluation_index = master_index[(master_index >= eval_date) & (master_index <= eval_end)]
        index_iterator[i] = {
            'optimizationIndex': optimization_index,
            'evaluationIndex': evaluation_index
        }

    return index_iterator

def get_annualization_factor(index: pd.DatetimeIndex) -> int:
    """
    Determine annualization factor based on frequency of a datetime index.

    Returns:
        int: Annualization factor (e.g., 260 for daily, 12 for monthly).
    """
    inferred = pd.infer_freq(index)
    if inferred is None:
        raise ValueError("Could not infer frequency from index.")

    inferred = inferred.lower()
    if inferred in ['d', 'b']:  # 'b' = business day
        return 260
    elif 'm' in inferred:
        return 12
    elif 'w' in inferred:
        return 52
    else:
        raise ValueError(f"Unsupported frequency: {inferred}")
    
def prepare_returns(prices: pd.DataFrame, frequency: str = "daily") -> pd.DataFrame:
    """
    Prepare returns from price data at specified frequency.

    Args:
        prices (pd.DataFrame): Daily price data
        frequency (str): "daily" or "monthly"

    Returns:
        pd.DataFrame: Cleaned returns at desired frequency
    """
    if frequency == "monthly":
        prices = prices.resample("ME").last()
    elif frequency == "weekly":
        prices = prices.resample("W").last()
    elif frequency == "daily":
        pass
    else:
        raise ValueError("Unsupported frequency: choose 'daily' or 'monthly'")

    returns = prices.pct_change()
    returns[prices.isna()] = None
    returns.replace([np.inf, -np.inf, -1, np.nan], 0, inplace=True)
    return returns

def colorize(string, color):
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "white": "\033[37m",
        "reset": "\033[0m"
    }
    color = colors.get(color, colors['reset'])
    return f"{color}{string}\033[0m"

def preload_bandwidth_csv(std=False):
    base_dir = f"data/kde_bandwidths_std/{settings.index_name}" if std else f"data/kde_bandwidths/{settings.index_name}"
    filename = os.path.join(base_dir, f"{settings.data_frequency}_{settings.window_size}_{settings.window_unit}.csv")
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        return pd.DataFrame(columns=settings.global_tickers)
    
def update_portfolio(index_history_df: pd.DataFrame, date: pd.Timestamp, N: int, seed: int, prev_tickers=None, maximum: bool = False, valid_tickers=None):
    """
    Update a portfolio by retaining valid tickers and replacing churned ones.

    Parameters:
    - index_history_df: DataFrame indexed by date, with column ['TICKERS'] (lists of tickers)
    - date: target date (pd.Timestamp or str)
    - N: number of tickers in the portfolio
    - seed: base seed for deterministic re-selection
    - prev_tickers: list of tickers from the previous period (optional, only None at t0)

    Returns:
    - List of N tickers for the new portfolio
    """
    date = pd.to_datetime(date)

    valid_rows = index_history_df.loc[index_history_df.index <= date]
    if valid_rows.empty:
        raise ValueError(f"No index composition available on or before {date}")

    current_tickers = set(valid_rows.iloc[-1]['TICKERS']) & valid_tickers

    if maximum:
        return sorted(current_tickers)

    if prev_tickers is None:
        if len(current_tickers) < N:
            raise ValueError(f"Not enough tickers in index on {date} to sample {N}")
        rng = random.Random(seed)
        return rng.sample(sorted(current_tickers), N)

    retained = [t for t in prev_tickers if t in current_tickers]
    n_to_replace = N - len(retained)

    if n_to_replace == 0:
        return retained

    candidates = sorted(current_tickers - set(retained))
    if len(candidates) < n_to_replace:
        raise ValueError(f"Not enough tickers to replace dropped ones on {date}")

    rng = random.Random(seed)
    new_picks = rng.sample(candidates, n_to_replace)

    return retained + new_picks
    
    