# Chronotype asymmetry arises from stochastic coupling between homeostatic and circadian processes near Arnold-tongue boundaries

Nguyen Trong Nguyen, Vuong Hung Truong, Jihwan Myung*

Laboratory of Braintime, Taipei Medical University

## Overview

This repository contains the simulation code and data for reproducing all figures in the accompanying manuscript. We show that the right-skewed distribution of human chronotype — observed worldwide but opposite in direction to the left-skewed distribution of intrinsic circadian periods — arises from stochastic variability in homeostatic sleep pressure dynamics within the Borbély two-process model framework.

The core model extends the two-process model by:
1. Formulating Process S build-up as a **drift-diffusion (random walk)** process, making sleep onset a first-passage time problem
2. Modeling **hierarchical entrainment**: sleep–wake → circadian clock → light–dark cycle
3. Mapping **Arnold tongue** structure in the circadian amplitude × sleep period parameter space

## Repository structure

```
chronotype-skewness/
│
├── src/
│   ├── twoprocess.py               # Core model: RK4 circadian solver, stochastic
│   │                                #   Process S, threshold crossing, chronotype extraction
│   └── config.py                    # Base parameters (Table 1), distribution settings
│
├── notebooks/
│   ├── 01_main_simulation.ipynb     # Population simulation + Figs. 2A, 3A–C
│   ├── 02_arnold_tongue.ipynb       # Composite error sweep over τ_sw × A → Fig. 4A
│   └── 03_kl_landscape.ipynb        # KL divergence sweep over τ_sw × A → Fig. 4B
│                                    #   and skewness sweep over μ_a × ξ_s_a → Fig. 2B–C
│
├── data/
│   ├── empirical/
│   │   └── period.csv               # Duffy et al. (2011) circadian period distribution
│   │                                #   extracted via WebPlotDigitizer v5
│   └── results/
│       ├── grid_search_results_parallel_freeday.csv   # Err landscape (Fig. 4A)
│       ├── parameter_space_KLdiv.csv                  # KL landscape (Fig. 4B)
│       └── sleep_period.csv                           # Intrinsic sleep-wake period
│                                                      #   as f(Hu0, A) for Fig. 4 x-axis
│
├── README.md
├── requirements.txt
└── LICENSE
```

## File descriptions

### `src/twoprocess.py`
The core simulation engine. Contains:
- **`simulate_circadian_numba()`**: Explicit RK4 integration of the phase oscillator (Eq. 4) with Numba JIT compilation. Includes sigmoid light-gating and configurable entrainment/free-running windows.
- **`_simulate_homeostatic_process()`**: Process S dynamics — Brownian motion with drift during wake (Eq. 5), exponential decay during sleep (Eq. 6). Threshold crossings trigger state transitions.
- **`TwoProcessModel`**: Object-oriented wrapper combining Process C and Process S with asymmetric thresholds (Eq. 2).
- **`simulate_population_batch()`**: Parallelized batch simulation of *n* agents with individual parameter draws. Returns per-agent chronotype, sleep duration, and sleep-wake period.
- **`generate_synthetic_periods_from_csv()`**: Bootstrap KDE + inverse transform sampling to generate synthetic circadian period distributions from the empirical histogram (Duffy et al., 2011).

### `src/config.py`
Default parameter configuration matching Table 1 of the manuscript:
- Circadian: `A = 0.05`, `K_light = 0.1`, `α = 19.5·2π/24`, `φ_lag = 18·2π/24`
- Thresholds: `M_U = 0.71`, `M_L = 0.2` (lower amplitude = A/5)
- Homeostatic: `μ ~ SkewNorm(a=-5, loc=0.03, scale=0.001)`, `ξ_s ~ SkewNorm(a=5, loc=5.5, scale=0.5)`, `σ = 0.005`
- Protocol: 14 days total, light entrainment days 0–12, free-running assessment days 12–14

### `data/empirical/period.csv`
Circadian period histogram (Period in hours, Value = frequency count) digitized from Duffy et al. (2011) *Proc. Natl. Acad. Sci. U.S.A.* using WebPlotDigitizer v5. 12 bins from 23.5–24.6 h, mean ≈ 24.09 h.

### `data/results/`
Pre-computed parameter sweep results. Each row is one (Hu0, A) configuration evaluated over 400 simulated individuals:
- **`grid_search_results_parallel_freeday.csv`**: Composite error `Err` (Eq. 10), skewness, median chronotype, mean sleep duration, sleep-wake period, Pearson *r* and *p* for period–chronotype correlation
- **`parameter_space_KLdiv.csv`**: Same grid but scored by KL divergence `D_KL(P_emp || P_sim)` (Eq. 9) against empirical chronotype distributions from Fischer et al. (2017) and Roenneberg et al. (2007)
- **`sleep_period.csv`**: Intrinsic sleep-wake period τ_sw (average inter-sleep-onset interval at A = 0) for each (Hu0) configuration, used to convert the x-axis of Fig. 4 from Hu0 to biologically interpretable sleep-wake period

## Reproducing figures

### Quick start

```bash
git clone https://github.com/[username]/chronotype-skewness.git
cd chronotype-skewness
pip install -r requirements.txt
```

### Figure-to-code mapping

| Figure | Description | Source |
|--------|-------------|--------|
| Fig. 1A | Empirical period & chronotype distributions | External data (WebPlotDigitizer) |
| Fig. 1B | Four noise regimes (linear generative model) | `notebooks/01_main_simulation.ipynb` |
| Fig. 2A | Single-run sleep-wake trace (social jet-lag protocol) | `notebooks/01_main_simulation.ipynb` |
| Fig. 2B–C | KL divergence vs. μ-skew × ξ_s-skew | `notebooks/03_kl_landscape.ipynb` |
| Fig. 3A–C | Period–chronotype joint distribution, residuals | `notebooks/01_main_simulation.ipynb` |
| Fig. 4A | Composite error Arnold tongue (τ_sw × A) | `notebooks/02_arnold_tongue.ipynb` |
| Fig. 4B | KL divergence landscape (τ_sw × A) | `notebooks/03_kl_landscape.ipynb` |

### Runtime notes
- Population simulations (n = 400–1000 agents) run in **~1–5 minutes** per configuration on a 32-core machine with Numba JIT.
- Full parameter sweeps (2,601 grid points × 400 agents each) take **~2–6 hours**. Pre-computed results are provided in `data/results/`.
- First execution triggers Numba compilation (~10 s); subsequent runs use cached machine code.

## Requirements

```
numpy>=1.24
scipy>=1.10
pandas>=1.5
matplotlib>=3.6
numba>=0.57
seaborn>=0.12
joblib>=1.2
```

## Citation

If you use this code, please cite:

> Nguyen, N.T., Truong, V.H., & Myung, J. (2026). Chronotype asymmetry arises from stochastic sleep homeostasis under circadian entrainment. (submitted).

## License

Code: MIT License
Data: CC BY 4.0

## Contact

Jihwan Myung — jihwan@tmu.edu.tw
Laboratory of Braintime, Taipei Medical University
