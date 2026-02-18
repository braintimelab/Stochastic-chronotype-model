# ==============================================================================
# CONFIGURATION (DEFINE EVERYTHING HERE)
# ==============================================================================
import numpy as np
import os
CONFIG = {
    # --- Simulation Settings ---
    'n_cycles': 14,          # Total days to simulate
    'start': 0,          # Day to start light entrainment
    'end': 12,           # Day to end light entrainment
    'dt': 0.1,               # Time step (hours)
    'start_day': 12,          # Day to start analyzing
    'end_day':14,             # Day to end analyzing
    'sample_size': 1000,     # Total agents to simulate
    'plot_n_agents': 100,    # How many agents to show on the plot (subset)
    'n_workers': 32,          # Number of parallel threads (adjust based on your CPU)
    
    # ---  Base Parameters ---
    'bio_params': {
        'a': 0.05,           # Amplitude of circadian oscillator 0.05 works
        'lag': 18*2*np.pi/24,            # Light lag -6*2*np.pi/24 or 18*2*np.pi/24
        'alpha': 19.5*2*np.pi/24 ,          # Phase angle for circadian oscillator 4.5*2*np.pi/24  or 19.5*2*np.pi/24 
        'Hu0': 0.71,         # Upper threshold baseline
        'Hl0': 0.2,          # Lower threshold baseline
        # 'end_day': 12,        # Day to end light entrainment
        'K': 0.1,            # Coupling strength
        # Note: 'period', 'mu', 'xis' will be overwritten by the generator
        'sigma': 0.005       # Noise level
    },

    # --- Distributions (for Random Sampling) ---
    'dist_params': {
        'xis_loc': 5.5, 'xis_scale': 0.5, 'xis_a': 5,      # SkewNorm for Decay 'xis_loc': 5.5, 'xis_scale': 0.5, 'xis_a': 5,
        'mu_loc': 0.03, 'mu_scale': 0.001, 'mu_a': -5,    # SkewNorm for Buildup 'mu_loc': 0.03, 'mu_scale': 0.001, 'mu_a': -5
        'csv_filename': 'period.csv'                     # File for period distribution
    }
}

# --- Auto-Calculated Variables  ---
CONFIG['duration'] = 24 * CONFIG['n_cycles']
CONFIG['time_arr'] = np.arange(0, CONFIG['duration'], CONFIG['dt'])
CONFIG['csv_path'] = os.path.join(os.getcwd(), CONFIG['dist_params']['csv_filename'])
