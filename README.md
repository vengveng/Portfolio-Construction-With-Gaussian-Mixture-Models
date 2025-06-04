# Portfolio Optimization Thesis Repository

This repository contains all code, data, and documentation used in the accompanying master’s thesis on portfolio optimization. It has been organized so that anyone can reproduce results, inspect custom optimization routines, and generate all figures and tables directly.

---

## 📂 Repository Structure

```
.
├── __pycache__/                              ← Python bytecode (ignored by Git)
├── cleaner.py                                ← Utility to delete unwanted files
├── core/
│   └── portfolio_lib.c                       ← C implementation (optional speed-up)
├── data/                                     ← All input data
│   ├── index_histories/                      ← Historical index price CSVs
│   │   ├── FTSE100_HISTORY.csv
│   │   ├── HSI_HISTORY.csv
│   │   ├── S&P500_HISTORY.csv
│   │   └── STOXX50_HISTORY.csv
│   ├── indices.csv                           ← List of all indices used
│   ├── kde_bandwidths_std/                   ← Bandwidth CSVs used by KDE-based methods
│   │   ├── FTSE100/
│   │   ├── HSI/
│   │   ├── S&P500/
│   │   └── STOXX50/
│   ├── leavers_joiners/                       ← Lists of index constituents over time
│   │   ├── FTSE100.csv
│   │   ├── HSI.csv
│   │   ├── S&P500.csv
│   │   └── STOXX50.csv
│   ├── list_equity/                          ← Regional equity ticker lists
│   │   ├── equity_amer.csv
│   │   ├── equity_em.csv
│   │   ├── equity_eur.csv
│   │   └── equity_pac.csv
│   ├── master_index.csv                      ← Master index metadata
│   ├── master_tickers.csv                    ← Master ticker metadata
│   ├── median_returns_with_envelope/         ← “Envelope” CSVs (post-clean)
│   │   ├── FTSE100/
│   │   ├── HSI/
│   │   ├── S&P500/
│   │   └── STOXX50/
│   ├── MV_cleaned.csv                         ← Cleaned market‐value data
│   ├── MV.csv                                 ← Raw market‐value data
│   ├── portfolio_returns/                     ← “_Nmax_” optimization results (post-clean)
│   │   ├── FTSE100/
│   │   ├── HSI/
│   │   ├── S&P500/
│   │   └── STOXX50/
│   ├── processed_portfolio_returns/           ← Aggregated portfolio‐metrics CSVs
│   │   ├── portfolio_metrics_FTSE100.csv
│   │   ├── portfolio_metrics_HSI.csv
│   │   ├── portfolio_metrics_S&P500.csv
│   │   └── portfolio_metrics_STOXX50.csv
│   └── TRI_cleaned.csv                        ← Total return index data (cleaned)
├── lib/
│   └── portfolio_lib.so                       ← Compiled C library (optional)
├── optimization_tools.py                      ← Custom objective functions & Portfolio class
├── plots.ipynb                                ← Jupyter notebook to recreate all thesis figures/tables
├── report/                                    ← LaTeX source, images, tables, compiled PDF
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
└── single_config.py                           ← Example script: run optimization for one configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites

- **Python 3.10+**  
- **pip** (to install Python dependencies)  
- **LaTeX** (e.g., TeX Live) if you plan to recompile `report/Thesis.tex`  
- **GNU Make** (optional, if you want to automate compilation tasks)  
- **C compiler** (e.g., `gcc` or `clang`) if you wish to compile `core/portfolio_lib.c`  
- Recommended Python packages (install via `pip install -r requirements.txt`, if you add one):  
  - `numpy`  
  - `pandas`  
  - `scipy`  
  - `cvxpy` (+ a solver backend like ECOS or OSQP)  
  - `matplotlib`  
  - `scikit-learn`  
  - `pyportfolioopt` (if used)  
  - `mkl` or `openblas` (optional, for faster linear algebra)  

<details>
<summary><strong>Optional: Build the C library (for tangency‐portfolio speed‐ups)</strong></summary>

1. Navigate to the `core/` folder:
   ```bash
   cd core
   ```
2. Compile `portfolio_lib.c` into a shared object. For example, on macOS/Linux:
   ```bash
   gcc -fPIC -O3 -shared -o ../lib/portfolio_lib.so portfolio_lib.c
   ```
3. Ensure `lib/portfolio_lib.so` is on the dynamic library path (or place it in `lib/` and let Python find it).  

If the C library is missing or fails to compile, the code will still run more slowly—just omit it, and the Python‐only implementations (in `optimization_tools.py`) will be used instead.
</details>

---

### 2. Install Python Dependencies

From the project root:
```bash
python3 -m venv venv
source venv/bin/activate         # macOS/Linux
# .\venv\Scripts\activate       # Windows PowerShell
pip install --upgrade pip
pip install numpy pandas scipy cvxpy matplotlib scikit-learn
# + any other libraries your local setup requires
```

*(You may choose to provide a `requirements.txt` in future for one‐line installs.)*

---

## 🔧 Running Single‐Configuration Optimizations

The file `single_config.py` demonstrates how to run a portfolio optimization for one `(type, solver, data-frequency, estimation-window, rebalance-frequency, etc.)` configuration. By default, it:  
1. Loads historical returns from `data/` (using `leavers_joiners/`, `index_histories/`, etc.).  
2. Instantiates a `Portfolio` object from `optimization_tools.py`.  
3. Solves a specific objective (e.g., maximum‐Sharpe or equal‐risk‐contribution) over the entire sample.  
4. Saves the resulting weights/time‐series (or performance metrics) into a CSV under `data/portfolio_returns/<Index>/…`.

#### How to use `single_config.py`

1. **Open** `single_config.py` to see the top‐level argument definitions. You’ll find variables like:
   ```python
   INDEX      = "S&P500"        # or "FTSE100", "HSI", "STOXX50"
   METHOD     = "max_sharpe"     # see valid_types below
   SOLVER     = "cvxpy"          # must be in valid_solvers_class[METHOD]
   DATA_FREQ  = "monthly"        # "daily", "weekly", etc.
   WINDOW     = "3_years"        # estimation period (e.g., "3_years", "6_months", "10_years")
   REB_FREQ   = "annual"         # rebalancing frequency ("monthly", "annual", etc.)
   RISK_FREE  = 0.01             # annualized risk‐free rate
   SEED       = 0                # for reproducibility (if applicable)
   ```
2. **Adjust** any of these parameters at the top of the file, or modify the function call at the bottom:
   ```python
   if __name__ == "__main__":
       portfolio = Portfolio(
           index=INDEX,
           method=METHOD,
           solver=SOLVER,
           data_freq=DATA_FREQ,
           window=WINDOW,
           reb_freq=REB_FREQ,
           rf=RISK_FREE,
           seed=SEED,
           use_c_lib=True,          # set False if you don’t want C acceleration
           **kwargs_for_method      # e.g., {'gamma':1}, {'h':None}, etc.
       )
       portfolio.run_full_sample()
       portfolio.save_results()
   ```
3. **Run** it:
   ```bash
   python single_config.py
   ```
4. **Output** will be written to:
   ```
   data/portfolio_returns/<INDEX>/
     <DATA_FREQ>_<REB_FREQ>_<WINDOW>_Nmax_seed<SEED>.csv
   ```
   By default, it uses the entire available date range. You can modify `start_date`/`end_date` in code if desired.

---

## 🛠 “optimization_tools.py” Overview

All custom objective functions, constraints, and the main `Portfolio` class live in `optimization_tools.py`. You can inspect or extend them easily—design choices are grouped by consistent naming. Below is a quick summary:

### 1. Available Optimization Types (`valid_types`)

```python
valid_types = (
    'erc',                # Equal‐Risk Contribution
    'max_sharpe',         # Tangency portfolio
    'min_var',            # Minimum‐variance
    'equal',              # 1/N equal‐weights
    'kde',                # Kernel‐Density Estimation of returns
    'gmm',                # Gaussian Mixture Model
    'kde_max_sharpe',     # Max‐Sharpe with KDE input
    'gmm_max_sharpe',     # Max‐Sharpe with GMM input
    'moments_only',       # Moments‐only (mean/variance)
    'value',              # Minimum‐Value‐at‐Risk
    'markowitz'           # Classic Markowitz (mean‐variance)
)
```

These are ranked (first is “best,” last is “worst” in a sample comparison).

### 2. Solver Map (`solver_map`)

Depending on `method` × `solver`, `Portfolio` will dispatch to one of these internal methods:

```python
solver_map = {
    ('erc',            'cvxpy'):   self._fit_erc_CVXPY,
    ('erc',            'scipy'):   self._fit_erc_SCIPY,
    ('erc',            'ipopt'):   self._fit_erc_IPOPT,
    ('max_sharpe',     'cvxpy'):   self._fit_max_sharpe_CVXPY,
    ('max_sharpe',     'scipy'):   self._fit_max_sharpe_SCIPY,
    ('max_sharpe',     'ipopt'):   self._fit_max_sharpe_IPOPT,
    ('min_var',        'cvxpy'):   self._fit_min_var_CVXPY,
    ('min_var',        'scipy'):   self._fit_min_var_SCIPY,
    ('kde',            'cvxpy'):   self._fit_KDE_CVXPY,
    ('kde',            'scipy'):   self._fit_KDE_SCIPY,
    ('kde',            'ipopt'):   self._fit_KDE_IPOPT,
    ('gmm',            'ipopt'):   self._fit_GMM_IPOPT,
    ('gmm',            'cvxpy'):   self._fit_GMM_CVXPY,
    ('gmm',            'scipy'):   self._fit_GMM_SCIPY,
    ('kde_max_sharpe', 'cvxpy'):   self._fit_KDE_max_sharpe,
    ('kde_max_sharpe', 'scipy'):   self._fit_KDE_max_sharpe,
    ('kde_max_sharpe', 'ipopt'):   self._fit_KDE_max_sharpe,
    ('gmm_max_sharpe', 'cvxpy'):   self._fit_GMM_max_sharpe,
    ('gmm_max_sharpe', 'scipy'):   self._fit_GMM_max_sharpe,
    ('gmm_max_sharpe', 'ipopt'):   self._fit_GMM_max_sharpe,
    ('markowitz',      'cvxpy'):   self._fit_markowitz_CVXPY,
    ('markowitz',      'scipy'):   self._fit_markowitz_SCIPY,
}
```

### 3. Default Parameter Sets

If you want to customize KDE, GMM, or Markowitz settings, look at these defaults:

```python
kde_defaults = {
    'h': None, 
    'gamma': 1, 
    'matrix': 'isotropic', 
    'cv': 5
}

gmm_defaults = {
    'k': 3, 
    'gamma': 1, 
    'method': 'default', 
    'k_min': 1, 
    'k_max': 10, 
    'cv': 5, 
    'solver': 'ipopt'
}

markowitz_defaults = {
    'gamma': 1
}
```

You can override these by passing `**kwargs_for_method` to the `Portfolio` constructor (e.g., `Portfolio(..., k=5, gamma=2)` for a custom GMM).

### 4. Supported Solvers per Method

```python
valid_solvers_class = {
    'erc':             ['cvxpy', 'scipy', 'ipopt'],
    'equal':           ['cvxpy', 'scipy', 'ipopt'],
    'value':           ['cvxpy', 'scipy', 'ipopt'],
    'max_sharpe':      ['cvxpy', 'scipy', 'ipopt'],
    'min_var':         ['cvxpy', 'scipy'],
    'kde':             ['cvxpy', 'ipopt', 'scipy'],
    'gmm':             ['ipopt', 'cvxpy', 'scipy'],
    'kde_max_sharpe':  ['cvxpy', 'scipy', 'ipopt'],
    'gmm_max_sharpe':  ['cvxpy', 'scipy', 'ipopt'],
    'markowitz':       ['cvxpy', 'scipy']
}
```

If you select a solver not in the list for your chosen method, the code will raise an error.

---

## 📊 Reproduce All Figures & Tables

The Jupyter notebook `plots.ipynb` contains the code to:

1. **Load** cleaned data from `data/median_returns_with_envelope/…` and `data/portfolio_returns/…`.  
2. **Compute** all statistics (Sharpe ratios, drawdowns, VaR/ES comparisons, etc.).  
3. **Generate** every figure image (PNG) used in the thesis.  
4. **Generate** every LaTeX table (`.tex`) and save under `report/tables/`.  
5. **Compile** any intermediate CSVs if needed.

To run it:

1. Activate your virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Open the notebook in VS Code or Jupyter Lab:
   ```bash
   jupyter lab plots.ipynb
   ```
3. Run all cells.  
   - All images will automatically save to `report/images/`  
   - All LaTeX tables will appear under `report/tables/`  
   - If you have missing data or “file not found” errors, double-check that your `data/median_returns_with_envelope/…` and `data/portfolio_returns/…` folders match what the notebook expects.

4. If you wish to regenerate the PDF thesis with updated figures/tables, run (from the `report/` directory):
   ```bash
   pdflatex Thesis.tex
   bibtex Thesis       # or biber Thesis.bcf, depending on your setup
   pdflatex Thesis.tex
   pdflatex Thesis.tex
   ```
   This will produce an updated `Thesis.pdf` with all new plots and tables.

---

## 🔄 Cleaning Up Generated/Untracked Files

- **`cleaner.py`**: Utility script that removes any files outside your explicitly “keep” lists. (For example, after testing or generating new CSVs, you can prune the directory back to only the desired files.)  
- **`.gitignore`** (not shown above) should contain at least:
  ```
  __pycache__/
  *.pyc
  *.so
  .DS_Store
  venv/
  *.log
  ```
  so that temporary or compiled files are not accidentally committed. 

---

## ⚠️ Notes & Caveats

1. **C‐Accelerated Tangency (max_sharpe) Code**  
   - The file `core/portfolio_lib.c` provides a C routine to speed up certain gradient‐based solves for tangency portfolios. If you want to use it, compile it into `lib/portfolio_lib.so` (see instructions above). Otherwise, Python‐only solvers (SciPy or CVXPY) will be used by default, and everything still works (just a bit slower).

2. **Data Licensing & Citation**  
   - The raw index‐history CSVs in `data/index_histories/…` must be cited appropriately (based on your data source).  
   - If you redistribute results, please include this repository as a citation in your thesis or any derived work.

3. **“By‐Default” Behavior**  
   - `single_config.py` runs over the **entire** data sample span, from the earliest available history to the latest. To test on sub‐periods, modify `start_date`/`end_date` inside the script.  
   - `plots.ipynb` also assumes a contiguous date range for each index; if you add new data, be sure to re‐index or fill gaps as necessary.

4. **Extending & Customizing**  
   - To add a **new objective**, implement a method named `_fit_<your_method>_<solver>` in `optimization_tools.py`, and register it in `solver_map`.  
   - To tweak default hyperparameters (e.g., number of GMM components, KDE bandwidth), modify the relevant dictionary (e.g., `gmm_defaults`) or pass overrides when instantiating `Portfolio`.

---

## 📚 Citation

If you use this code or data in your work, please cite:

> **[Your Name], “Master’s Thesis on Portfolio Optimization,” [University Name], [Year].**  
> GitHub: `https://github.com/your‐username/your‐repo`  
> DOI: *[if assigned]*

---

Thank you for exploring this repository!