# Portfolio-Optimisation Thesis — Reproducibility Repository

A self-contained archive of the code, (subset of) data, and *LaTeX* sources
used in the master's thesis **"Portfolio Construction with Gaussian Mixture
Models"** (Ivan Khalin, June 2025).  
Everything required to regenerate tables, figures, and single-configuration
experiments is shipped in this repo; the 40 GB of raw return panels is left
out to keep the footprint reasonable.

---

## 1 Layout

```
.
├── core/                   # optional C backend (portfolio_lib.c)
├── lib/                    # .so library ends up here if you compile
├── data/                   # only the filtered CSVs needed for plots
├── report/                 # full thesis source, images, tables, PDF
├── optimisation_tools.py   # main objective + solver dispatcher
├── single_config.py        # run one configuration end-to-end
├── plots.ipynb             # regenerate every figure / table
└── requirements.txt
```

---

## 2 Quick Start & Objective Catalogue

### 2.1 Run a single configuration

Edit the header of `single_config.py` (index, γ, window, *N*, …) and run:

```bash
python single_config.py
```

Results print to screen; nothing is written to disk unless you uncomment `portfolio_returns.to_csv(...)`.

### 2.2 Available objective functions

| Keyword (portfolio_configs) | Purpose | Main solver(s) |
|---|---|---|
| `equal`, `value` | Benchmarks (1/N, value-weight) | — |
| `min_var` | Global minimum variance | CVXPY / SciPy |
| `max_sharpe` | Classical tangency (max-SR) | CVXPY / IPOPT / SciPy |
| `erc` | Equal-risk-contribution | CVXPY / IPOPT |
| `markowitz` | Mean–variance with γ | CVXPY / SciPy |
| `kde` | Proposed KDE exp-utility with γ | CVXPY / IPOPT / SciPy |
| `kde_max_sharpe` | Tangency under KDE moments | CVXPY / IPOPT |
| `gmm` | GMM exp-utility with γ | IPOPT / CVXPY / SciPy |
| `gmm_max_sharpe` | Tangency under GMM moments | CVXPY / IPOPT |

**Bandwidth vs. clusters**
- **KDE**: `h` (or matrix) is the kernel bandwidth; if a scalar is provided ⇒ isotropic, diagonal otherwise.
- **GMM**: `k` is the number of Gaussian clusters to be used (default is 8); cross-validation can be used with GMM class methods.

### 2.3 Default hyper-parameters

```python
kde_defaults = {'h': None, 'gamma': 1, 'matrix': 'isotropic', 'cv': 5}
gmm_defaults = {'k': 3, 'gamma': 1, 'k_min': 1, 'k_max': 10, 'cv': 5}
markowitz_defaults = {'gamma': 1}
```

### 2.4 Solver map

These are the avilible solvers for each of the portfolio types; 'best' solvers based on our testing of speed and optimality.
If a solver fails, `Portfolio` will attempt the next one in the list.

```python
valid_solvers_class = {
    'min_var'        : ['cvxpy', 'scipy'],
    'max_sharpe'     : ['cvxpy', 'ipopt', 'scipy'],
    'erc'            : ['cvxpy', 'ipopt', 'scipy'],
    'kde'            : ['cvxpy', 'ipopt', 'scipy'],
    'gmm'            : ['ipopt', 'cvxpy', 'scipy'],
    'kde_max_sharpe' : ['cvxpy', 'ipopt', 'scipy'],
    'gmm_max_sharpe' : ['cvxpy', 'ipopt', 'scipy'],
    'markowitz'      : ['cvxpy', 'scipy'],
}
```

If `lib/portfolio_lib.so` is present and properly linked,
tangency-portfolio optimisation (`max_sharpe`, `kde_max_sharpe`, …) switches to
the C gradient for a sizeable speed-up; otherwise it falls back on the pure
Python path automatically.

---

## 3 Regenerating All Tables & Figures

Open `plots.ipynb` and run all cells.
The notebook
1. loads the slimmed-down CSVs under `data/`,
2. recomputes performance statistics,
3. outputs every plot to `report/images/`,
4. rewrites the accompanying LaTeX tables in `report/tables/`.

---

## 4 Optional C Acceleration

`core/portfolio_lib.c` contains hand-optimised gradients for the tangency
objectives. Build instructions are provided in the file.
If the `.so` cannot be compiled, everything still runs.

---

## 5 Notes

- **Storage** – Full raw return matrices (~40 GB) are not in the repo. Paths in the notebook assume only the filtered CSVs included here.

---

## Acknowledgements

If you use or adapt this code please cite the thesis:

Ivan Khalin, "Portfolio Construction with Gaussian Mixture Models,"
Master's thesis, HEC Lausanne, June 2025.

---