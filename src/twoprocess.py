import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, gaussian_kde
from concurrent.futures import ThreadPoolExecutor
import os
from numba import njit, prange
from config import CONFIG

# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================

# --- Global Helpers for ODE Solver ---
@njit
def phase_light(t, end, start, light_lag):
    """Calculates the phase of the light signal."""
    if (t < 24 * end) and (t >= 24 * start):
        return (2 * np.pi / 24) * t + light_lag
    else:
        return 0.0

@njit
def thres(t, threshold, end):
    """Calculates the threshold for light gating."""
    return threshold if t < 24 * end else 20.0

@njit
def tf(x, th):
    """Transfer function (sigmoid gating)."""
    return 1.0 / (1.0 + np.exp(-10.0 * (x - th)))

# --- NUMBA OPTIMIZED SOLVERS ---

@njit
def get_dphi(t, phi, omega, K, end, start, light_lag, threshold):
    """
    Calculates the derivative dphi/dt and auxiliary values.
    Used by the RK4 solver.
    """
    pl = phase_light(t, end, start, light_lag)
    th = thres(t, threshold, end)
    g = tf(np.sin(pl), th)
    dphi = omega + K * g * np.sin(pl - phi)
    return dphi, pl, g

@njit
def simulate_circadian_numba(duration, dt, phi0, omega, K, end, start, light_lag, threshold):
    """
    Runs the Circadian Process simulation using an explicit RK4 solver.
    Replaces scipy.integrate.solve_ivp for massive speedup.
    """
    n_steps = int(np.ceil(duration / dt))
    
    # Pre-allocate arrays
    ts = np.empty(n_steps)
    phi_vals = np.empty(n_steps)
    phi_light_vals = np.empty(n_steps)
    gating_vals = np.empty(n_steps)
    
    t = 0.0
    phi = phi0
    
    for i in range(n_steps):
        # Store State
        ts[i] = t
        phi_vals[i] = phi
        
        # Calculate current aux values for storage
        # (We re-calculate light/gating here to ensure 'phi_light_vals' matches 'ts[i]')
        dphi_curr, pl_curr, g_curr = get_dphi(t, phi, omega, K, end, start, light_lag, threshold)
        phi_light_vals[i] = pl_curr
        gating_vals[i] = g_curr
        
        # --- RK4 Integration Step ---
        
        # k1
        k1 = dphi_curr
        
        # k2
        dphi_k2, _, _ = get_dphi(t + 0.5*dt, phi + 0.5*dt*k1, omega, K, end, start, light_lag, threshold)
        k2 = dphi_k2
        
        # k3
        dphi_k3, _, _ = get_dphi(t + 0.5*dt, phi + 0.5*dt*k2, omega, K, end, start, light_lag, threshold)
        k3 = dphi_k3
        
        # k4
        dphi_k4, _, _ = get_dphi(t + dt, phi + dt*k3, omega, K, end, start, light_lag, threshold)
        k4 = dphi_k4
        
        # Update
        phi += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        
    return ts, phi_vals, phi_light_vals, gating_vals

@njit
def _simulate_homeostatic_process(ts, upper, lower, y0, first_awake, mu, sigma, xis):
    """Simulate the homeostatic process with drift-diffusion and exponential decay."""
    n = len(ts)
    H = np.zeros(n)
    awake = np.zeros(n, dtype=np.bool_)
    H[0] = y0
    awake[0] = first_awake
    
    curr_H = y0
    is_awake = first_awake
    
    # Pre-calculate dt to save time in loop (assuming constant dt)
    dt_s = ts[1] - ts[0]
    sqrt_dt = np.sqrt(dt_s)
    
    for i in range(1, n):
        if is_awake:
            # Drift diffusion
            noise = np.random.normal(0.0, 1.0)
            curr_H = curr_H + mu * dt_s + sqrt_dt * sigma * noise
            
            if curr_H >= upper[i]:
                is_awake = False
        else:
            # Exponential decay
            curr_H = curr_H * np.exp(-dt_s / xis)
            
            if curr_H <= lower[i]:
                is_awake = True
        
        H[i] = curr_H
        awake[i] = is_awake
    
    return H, awake

class TwoProcessModel:
    def __init__(self, params):
        self.params = params.copy()
        # Initialize Clock State
        omega = params.get('omega', (2 * np.pi) / params['period'])
        
        # Unpack arguments for Numba function
        self.omega = omega
        self.K = params.get('K', 0.1)
        self.end = params.get('end', 12)
        self.start = params.get('start', 0)
        self.lag = params.get('lag', 16*np.pi/12)
        self.threshold = params.get('threshold', 0)
        
        self.phi0 = 0.0
        self.alpha = params.get('alpha', 0)
        
        # Homeostat Params
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.xis = params['Xis'] 
        self.hu0 = params['Hu0']
        self.hl0 = params['Hl0']
        self.amp = params['a']
        
        self.results = None

    def simulate(self, duration, dt, y0, first_awake=True):
        # 1. Circadian Process (Fast Numba RK4)
        ts, phi_vals, phi_light_vals, gating = simulate_circadian_numba(
            duration, dt, self.phi0, self.omega, self.K, self.end, self.start, self.lag, self.threshold
        )

        circ_signal = phi_vals + self.alpha 

        # 2. Thresholds (Vectorized Numpy)
        sin_circ = np.sin(circ_signal)
        upper = self.hu0 + self.amp * sin_circ
        lower = self.hl0 + (self.amp/5.0) * sin_circ
        
        # 3. Homeostatic Process (Fast Numba Loop)
        H, awake = _simulate_homeostatic_process(ts, upper, lower, y0, first_awake, self.mu, self.sigma, self.xis)
                
        self.results = pd.DataFrame({
            'time': ts, 'H': H, 'awake': awake, 
            'upper': upper, 'lower': lower, 'phi_light_vals': phi_light_vals, 'gating': gating
        })
        return self.results

    def get_chronotypes(self):
        if self.results is None: return pd.DataFrame()
        df = self.results
        changes = df['awake'].astype(int).diff()
        sleeps = df.loc[changes == -1, 'time'].values
        wakes = df.loc[changes == 1, 'time'].values
        
        if len(wakes) > 0 and len(sleeps) > 0:
            if wakes[0] < sleeps[0]: wakes = wakes[1:]
            n = min(len(sleeps), len(wakes))
            sleeps, wakes = sleeps[:n], wakes[:n]
            return pd.DataFrame({
                'Sleep_Onset': sleeps,
                'Wake_Onset': wakes,
                'Chronotype': (sleeps%24 + (wakes-sleeps)/2) % 24
            })
        return pd.DataFrame()

# ==============================================================================
# GENERATION & WORKER FUNCTIONS
# ==============================================================================


# --- NUMBA OPTIMIZED KERNEL DENSITY ESTIMATION ---

@njit
def _kde_evaluate_numba(data, grid, bw):
    """
    Evaluates Gaussian KDE on a grid for a single dataset.
    Replaces scipy.stats.gaussian_kde.evaluate
    """
    n = len(data)
    m = len(grid)
    pdf = np.zeros(m)
    
    # Precompute constants
    norm_factor = 1.0 / (np.sqrt(2.0 * np.pi) * n * bw)
    inv_bw = 1.0 / bw
    
    for i in range(m):
        x = grid[i]
        sum_kernel = 0.0
        # Unroll or simple loop - Numba vectorizes this well
        for j in range(n):
            z = (x - data[j]) * inv_bw
            sum_kernel += np.exp(-0.5 * z * z)
        pdf[i] = sum_kernel * norm_factor
        
    return pdf

@njit(parallel=True)
def _run_bootstrap_numba(periods, weights, x_grid, n_original, jitter_range, iters):
    """
    Runs the entire bootstrap loop in parallel.
    Replaces the Python loop that called scipy.stats.gaussian_kde 1000 times.
    """
    n_grid = len(x_grid)
    boot_results = np.zeros((iters, n_grid))
    
    # Prepare CDF for weighted sampling (Inverse Transform Sampling)
    w_cdf = np.cumsum(weights)
    w_cdf = w_cdf / w_cdf[-1]  # Normalize
    
    # Parallel Loop over Bootstrap Iterations
    for i in prange(iters):
        # 1. Weighted Resampling
        # Generate random indices based on weights using searchsorted (Binary Search)
        rand_u = np.random.random(n_original)
        indices = np.searchsorted(w_cdf, rand_u)
        
        # Construct the resampled array
        resampled = np.empty(n_original)
        for k in range(n_original):
            resampled[k] = periods[indices[k]]
            
        # 2. Add Jitter (Uniform)
        noise = np.random.uniform(-jitter_range, jitter_range, n_original)
        sample = resampled + noise
        
        # 3. Calculate Bandwidth (Scott's Rule: n**(-1/5) * std)
        # Note: Scipy uses n**(-1./(d+4)), where d=1 -> n**(-0.2)
        std_dev = np.std(sample)
        if std_dev == 0:
            bw = 1.0 # Fallback
        else:
            bw = std_dev * (n_original ** (-0.2))
        
        # 4. Evaluate KDE
        boot_results[i, :] = _kde_evaluate_numba(sample, x_grid, bw)
        
    return boot_results

# --- PYTHON WRAPPER FUNCTIONS ---

def generate_synthetic_periods_from_csv(csv_path, target_n=1000, bootstrap_iters=1000, seed=1):
    """
    Refined Pipeline (Numba Accelerated): 
    1. Loads Period/Value counts.
    2. Runs Parallel Numba Bootstrap KDE.
    3. Performs Inverse Transform Sampling.
    """
    # Load distribution
    df_dist = pd.read_csv(csv_path)
    periods = df_dist['Period'].values.astype(np.float64) # Ensure float for Numba
    counts = df_dist['Value'].values.astype(np.float64)
    
    np.random.seed(seed)
    
    # Setup Logic
    n_original = int(np.sum(counts))
    weights = counts / np.sum(counts)
    
    # Grid setup
    x_grid = np.linspace(np.min(periods) - 0.3, np.max(periods) + 0.3, 1000)
    bin_width = np.mean(np.diff(periods))
    jitter_range = bin_width / 2
    
    # --- STEP 1: Fast Bootstrap (Numba) ---
    # We pass the arrays to the compiled function
    boot_results = _run_bootstrap_numba(
        periods, weights, x_grid, n_original, jitter_range, bootstrap_iters
    )

    # Step 2: Extract Median PDF
    median_pdf = np.percentile(boot_results, 50, axis=0)
    
    # Step 3: Inverse Transform Sampling (PDF -> CDF -> Sample)
    dx = x_grid[1] - x_grid[0]
    cdf = (median_pdf * dx).cumsum()
    cdf /= cdf.max()  
    
    # Generate the 1D array
    u = np.random.uniform(0, 1, target_n)
    synthetic_sample = np.interp(u, cdf, x_grid)
    
    return synthetic_sample

def generate_population_params(config):
    """Generates the list of parameter dictionaries based on CONFIG."""
    print("Generating population parameters...")
    n = config['sample_size']
    d_params = config['dist_params']
    
    # 1. Periods (Accelerated)
    periods = generate_synthetic_periods_from_csv(config['csv_path'], target_n=n, seed=1)
    
    # 2. Biological Params (SkewNorm is efficient in Scipy C-backend, keeping as is)
    xis = skewnorm.rvs(a=d_params['xis_a'], loc=d_params['xis_loc'], scale=d_params['xis_scale'], size=n)
    # xis = np.repeat (d_params['xis_loc'], n)
    mus = skewnorm.rvs(a=d_params['mu_a'], loc=d_params['mu_loc'], scale=d_params['mu_scale'], size=n)
    # mus = np.repeat(d_params['mu_loc'], n)
    sigmas = np.repeat(config['bio_params']['sigma'], n)
    
    # Shuffle
    for arr in [periods, xis, mus, sigmas]: np.random.shuffle(arr)

    # 3. Pack into list of dicts (Optimized List Comprehension)
    base = config['bio_params']
    
    # Pre-create the list to avoid repetitive .update() calls
    pop_list = [
        {**base, 'period': periods[i], 'Xis': xis[i], 'mu': mus[i], 'sigma': sigmas[i]}
        for i in range(n)
    ]
        
    return pop_list

def worker_simulation(agent_params):
    """Function passed to ThreadPoolExecutor."""
    model = TwoProcessModel(agent_params)
    
    # Simulate
    df_res = model.simulate(
        duration=CONFIG['duration'], 
        dt=CONFIG['dt'], 
        y0=agent_params['Hl0'], 
        first_awake=True
    )
    
    # Analyze
    df_stats = model.get_chronotypes()
    
    # Filter Stats for relevant days
    if not df_stats.empty:
        mask = (df_stats['Sleep_Onset'] <= CONFIG['duration']) & \
               (df_stats['Sleep_Onset'] >= 24 * CONFIG['start_day'])
        df_stats = df_stats.loc[mask]
        
    return df_res, df_stats
@njit
def get_combined_stats_windowed(awake, ts, start_hour, end_hour):
    """
    Calculates three metrics in one go for events falling within the window:
    1. Chronotype (Mid-sleep point)
    2. Sleep Duration (Sleep -> Wake)
    3. Sleep Period (Sleep Onset -> Next Sleep Onset)
    
    Returns: (avg_chrono, avg_duration, avg_period)
    """
    n_steps = len(awake)
    wakes = []
    sleeps = []
    
    # 1. Detect All Transitions
    for i in range(1, n_steps):
        if not awake[i-1] and awake[i]:
            wakes.append(ts[i])
        if awake[i-1] and not awake[i]:
            sleeps.append(ts[i])

    # --- Metrics 1 & 2: Chronotype and Duration ---
    # Logic: Pair Sleep -> Wake. If Wake is in window, count it.
    
    chrono_sum = 0.0
    dur_sum = 0.0
    count_cd = 0
    w_ptr = 0
    
    for s_idx in range(len(sleeps)):
        s_time = sleeps[s_idx]
        
        # Find next Wake
        found_wake = -1.0
        while w_ptr < len(wakes):
            if wakes[w_ptr] > s_time:
                found_wake = wakes[w_ptr]
                break
            w_ptr += 1
            
        if found_wake != -1.0:
            # Check Window (Based on Wake Time)
            if found_wake >= start_hour and found_wake <= end_hour:
                dur = found_wake - s_time
                if 0 < dur < 24:
                    mid = (s_time + dur * 0.5) % 24
                    chrono_sum += mid
                    dur_sum += dur
                    count_cd += 1

    # --- Metric 3: Sleep Period ---
    # Logic: Diff between consecutive Sleep Onsets. If Current Sleep is in window, count it.
    
    period_sum = 0.0
    count_p = 0
    
    if len(sleeps) >= 2:
        for k in range(1, len(sleeps)):
            current_onset = sleeps[k]
            prev_onset = sleeps[k-1]
            
            # Check Window (Based on Sleep Onset Time)
            if current_onset >= start_hour and current_onset <= end_hour:
                p_val = current_onset - prev_onset
                # Sanity Filter (reject microsleeps/noise)
                period_sum += p_val
                count_p += 1
    
    # --- Final Averages ---
    avg_c = chrono_sum / count_cd if count_cd > 0 else np.nan
    avg_d = dur_sum / count_cd if count_cd > 0 else np.nan
    avg_p = period_sum / count_p if count_p > 0 else np.nan
    
    return avg_c, avg_d, avg_p


# --- 2. Batch Simulation Function (The Optimizer) ---
@njit(parallel=True)
def simulate_population_batch(
    n_agents, duration, dt, 
    periods, xis, mus, sigmas, 
    Hu0, Hl0, K, a, 
    end, start, lag, threshold, alpha,
    start_day_analysis, end_day_analysis 
):
    # Output: [Mean_Chrono, Mean_Duration, Mean_Period]
    results = np.empty((n_agents, 3))
    
    start_hour = (start_day_analysis - 0.5) * 24.0
    end_hour = (end_day_analysis + 1) * 24.0
    
    for i in prange(n_agents):
        tau = periods[i]
        omega = (2 * np.pi) / tau
        
        # 1. Circadian
        ts, phi_vals, _, _ = simulate_circadian_numba(
            duration, dt, 0.0, omega, K, end, start, lag, threshold
        )
        
        # 2. Homeostatic
        circ_signal = phi_vals + alpha
        sin_circ = np.sin(circ_signal)
        upper = Hu0 + a * sin_circ
        lower = Hl0 + (a/5.0) * sin_circ
        
        _, awake = _simulate_homeostatic_process(
            ts, upper, lower, Hl0, True, mus[i], sigmas[i], xis[i]
        )
        
        # 3. Get All Stats
        c_avg, d_avg, p_avg = get_combined_stats_windowed(awake, ts, start_hour, end_hour)
        
        results[i, 0] = c_avg
        results[i, 1] = d_avg
        results[i, 2] = p_avg
        
    return results