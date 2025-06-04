# Portfolio Optimization Thesis Repository

This repository contains all code, data, and documentation used in the accompanying master’s thesis on portfolio optimization. It is organized so that anyone can reproduce results, inspect custom optimization routines, and generate all figures and tables directly.

---

## 📂 Repository Structure

```
.
├── __pycache__/                     
├── cleaner.py                       
├── core/                            
│   └── portfolio_lib.c              
├── data/                            ← Contains only the files required to run optimizations and generate plots; the 40+ GB of raw portfolio-returns data is omitted.  
│   ├── index_histories/             
│   ├── indices.csv                  
│   ├── kde_bandwidths_std/          
│   ├── leavers_joiners/             
│   ├── list_equity/                 
│   ├── master_index.csv             
│   ├── master_tickers.csv           
│   ├── median_returns_with_envelope/ ← Only the CSVs that are ultimately selected by the plotting routines  
│   ├── MV_cleaned.csv               
│   ├── MV.csv                       
│   ├── portfolio_returns/            ← Only the “_Nmax_” CSVs needed for reproducing performance tables  
│   ├── processed_portfolio_returns/  
│   └── TRI_cleaned.csv              
├── lib/                             
│   └── portfolio_lib.so             
├── optimization_tools.py            
├── plots.ipynb                      
├── report/                          
│   ├── Appendices/                  
│   ├── Bibliography.bib             
│   ├── Chapters/                    
│   ├── images/                      
│   ├── Miscellaneous/               
│   ├── tables/                      
│   ├── Thesis.bbl                   
│   ├── Thesis.pdf                   
│   ├── Thesis.synctex.gz            
│   ├── Thesis.tex                   
│   └── trees/                       
├── single_config.py                 
└── requirements.txt                 
```

---

## 🚀 Getting Started

### 1. Prerequisites

* **Python 3.10+**
* A C compiler (e.g., `gcc` or `clang`) if you wish to compile the optional C library

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
cvxpy
cyipopt
matplotlib
numba
numpy
pandas
pathos
scikit-learn
scipy
seaborn
tqdm
```

<details>
<summary><strong>Optional: Build the C library (for tangency-portfolio speed-ups)</strong></summary>

If you wish to use the C-accelerated routines in `core/portfolio_lib.c`, compile it into `lib/portfolio_lib.so`. For example:

```bash
cd core
gcc -fPIC -O3 -shared -o ../lib/portfolio_lib.so portfolio_lib.c
```

If the C library is missing or fails to compile, the code will still run (more slowly) using the pure-Python implementations in `optimization_tools.py`.

</details>

---

## 🔧 Running Single-Configuration Optimizations

All settings for a single-configuration run are controlled at the top of `single_config.py`. By default, it runs over the entire data sample and does not save outputs unless you explicitly call the save methods.

```python
# Configuration block of a single configuration
# -------------------------------------------------------------------------------
indices = ['FTSE100', 'HSI', 'S&P500', 'STOXX50']
index_name = indices[2]  # Change index here (e.g., 'S&P500')
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
# -------------------------------------------------------------------------------
```

The `Portfolio` constructor (in `optimization_tools.py`) automatically dispatches to one of the available methods based on the `name` and solver. If you choose a method or solver that isn’t compatible, the code will raise an error. Available portfolio types include:

```python
valid_types = (
    'erc', 'max_sharpe', 'min_var', 'equal', 'kde', 'gmm',
    'kde_max_sharpe', 'gmm_max_sharpe', 'moments_only',
    'value', 'markowitz'
)
```

Default hyperparameter dictionaries:

```python
kde_defaults = {'h': None, 'gamma': 1, 'matrix': 'isotropic', 'cv': 5}
gmm_defaults = {'k': 3, 'gamma': 1, 'method': 'default', 'k_min': 1, 'k_max': 10, 'cv': 5, 'solver': 'ipopt'}
markowitz_defaults = {'gamma': 1}
```

Supported solvers per method:

```python
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
```

To run, simply call:

```bash
python single_config.py
```

Outputs are not saved by default; you must invoke the specific save methods on the `Portfolio` instance if you wish to persist results to CSV.

---

## 🛠 “optimization\_tools.py” Overview

`optimization_tools.py` houses all custom objective functions and the main `Portfolio` class. Its key features:

1. **Portfolio types (`valid_types`)**:

   ```python
   ('erc', 'max_sharpe', 'min_var', 'equal', 'kde', 'gmm',
    'kde_max_sharpe', 'gmm_max_sharpe', 'moments_only',
    'value', 'markowitz')
   ```
2. **Default hyperparameters**:

   ```python
   kde_defaults = {'h': None, 'gamma': 1, 'matrix': 'isotropic', 'cv': 5}
   gmm_defaults = {'k': 3, 'gamma': 1, 'method': 'default', 'k_min': 1, 'k_max': 10, 'cv': 5, 'solver': 'ipopt'}
   markowitz_defaults = {'gamma': 1}
   ```
3. **Solver dispatch (`solver_map`)**:
   Each (`method`, `solver`) pair maps to an internal `_fit_*` routine.
4. **Supported solvers per method (`valid_solvers_class`)**.
5. **C-accelerated routines**: If `use_c_lib=True` and `lib/portfolio_lib.so` is present, tangency-portfolio solves run via the compiled C code.

---

## 📊 Reproduce All Figures & Tables

The Jupyter notebook `plots.ipynb` automatically:

1. Loads cleaned data from `data/median_returns_with_envelope/…` and `data/portfolio_returns/…`.
2. Computes statistics (Sharpe ratios, drawdowns, VaR/ES, etc.).
3. Generates each figure (saved under `report/images/`).
4. Generates each LaTeX table (saved under `report/tables/`).

To run it, simply open in Jupyter Lab or VS Code and execute all cells. Ensure that the data folders match what the notebook expects; files that meet the selection criteria are already included in this repo.

---

## 🔄 Cleaning Up Generated/Untracked Files

* **`cleaner.py`**: Utility to remove any files outside the explicitly “keep” lists. For example, after generating new CSVs or temporary artifacts, it prunes directories back to only the files needed for reproducing results and plots.

* **`.gitignore`** (in repo root) ensures that the following are not tracked:

  ```
  __pycache__/
  *.pyc
  *.so
  .DS_Store
  venv/
  *.log
  ```

---

## ⚠️ Notes & Caveats

1. **C-Accelerated Tangency (max\_sharpe) Code**

   * If you compile `core/portfolio_lib.c` into `lib/portfolio_lib.so`, the gradient‐based solves for tangency portfolios run much faster. Otherwise, Python-only solvers (SciPy or CVXPY) are used.

2. **Data Licensing & Citation**

   * The raw index histories under `data/index_histories/…` must be cited appropriately (according to your data source).
   * Please cite this repository in your thesis or any derived work.

3. **Data Contents**

   * The `data/` folder includes only the subsets required to run optimizations and generate the final plots/tables. The full raw portfolio-returns (40+ GB) is intentionally omitted.
   * Files under `median_returns_with_envelope/…` and `portfolio_returns/…` are precisely those that meet the selection criteria used by `plots.ipynb`.

4. **Extending & Customizing**

   * To add a new objective, implement a `_fit_<your_method>_<solver>` method in `optimization_tools.py` and register it in `solver_map`.
   * To tweak default hyperparameters (e.g., number of GMM components, KDE bandwidth), modify the relevant default dictionary or pass overrides when constructing a `Portfolio` instance.

---

## 📚 Citation

If you use this code or data in your work, please cite:

> **\[Your Name], “Master’s Thesis on Portfolio Optimization,” \[University Name], \[Year].**
> GitHub: `https://github.com/your-username/your-repo`
> DOI: *\[if assigned]*

---

Thank you for exploring this repository! Feel free to open an issue or send a pull request if you spot bugs, wish to suggest improvements, or add new optimization routines.
