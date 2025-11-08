import numpy as np
from antropy import app_entropy
from typing import Union, Tuple

# --- Constants for Entropy Calculation ---
# Embedding dimension (m) is typically 2
EMBEDDING_DIM = 2 
# Time delay (tau) is typically 1 for continuous signals
DELAY = 1 
# Tolerance (r) is usually 0.2 * Standard Deviation (SD) of the signal
# This value is handled by the functions internally by default or based on input

def approximate_entropy(signal: np.ndarray, m: int = EMBEDDING_DIM, r_factor: float = 0.2) -> float:
    """
    Calculates the Approximate Entropy (ApEn) of a 1D signal.
    
    ApEn measures the complexity and unpredictability of a time-series.
    We use the robust implementation from the 'antropy' library.
    
    Args:
        signal (np.ndarray): The time-series signal (e.g., a single EEG channel epoch).
        m (int): Embedding dimension.
        r_factor (float): Multiplier for standard deviation to determine tolerance (r).

    Returns:
        float: The Approximate Entropy value.
    """
    if len(signal) < 20:
        return np.nan # Cannot reliably compute ApEn on very short segments

    # ApEn requires the signal to be a 1D array
    r = r_factor * np.std(signal, ddof=1)
    
    # We use antropy's implementation which is fast and reliable.
    try:
        apen = app_entropy(signal, m=m, r=r)
        return float(apen)
    except Exception as e:
        print(f"Error calculating ApEn: {e}")
        return np.nan

def fuzzy_entropy(signal: np.ndarray, m: int = EMBEDDING_DIM, n: int = 2, r_factor: float = 0.2) -> float:
    """
    Calculates the Fuzzy Entropy (FuzzyEn) of a 1D signal.
    
    FuzzyEn is an improvement over ApEn/SampEn using fuzzy membership functions
    to define vector similarity, making it more robust to noise and short data.
    
    Args:
        signal (np.ndarray): The time-series signal.
        m (int): Embedding dimension.
        n (int): Fuzzy power (exponent in the exponential membership function).
        r_factor (float): Multiplier for standard deviation to determine tolerance (r).

    Returns:
        float: The Fuzzy Entropy value.
    """
    N = len(signal)
    if N < 20:
        return np.nan

    r = r_factor * np.std(signal, ddof=1)

    def phi(m_dim):
        """Calculates the correlation sum component Phi_m(r, n)"""
        # 1. Baseline removal and vector embedding
        X = np.array([signal[i:i + m_dim] for i in range(N - m_dim + 1)])
        
        # Calculate mean of each embedded vector (x_0_i)
        X0 = np.mean(X, axis=1)
        
        # Subtract the mean from each vector
        X_centered = X - X0[:, None] 

        # 2. Calculate distance matrix (Chebyshev distance, d_ij)
        # We use broadcasted array operations for speed
        dists = np.max(np.abs(X_centered[:, None] - X_centered[None, :]), axis=2)
        
        # 3. Calculate fuzzy similarity (D_ij) using exponential function
        # This is the core fuzzy step: exp(-(d_ij)^n / r)
        D = np.exp(-(dists**n) / r)
        
        # 4. Calculate Phi_m (excluding self-matches, i != j)
        sum_D = np.sum(D, axis=1) # Sum of similarity for each vector
        
        # The sum includes D_ii=1 (for i=j). We exclude D_ii=1 by subtracting 1.
        # N - m_dim is the total number of vectors
        # N - m_dim - 1 is the number of pairs for each vector (j != i)
        
        # Sum of D_ij for j != i
        sum_no_self = sum_D - 1.0 
        
        # C_i is sum_no_self / (N - m_dim - 1)
        C = sum_no_self / (N - m_dim - 1)
        
        # Phi_m is the average of C_i
        Phi = np.mean(C)
        return Phi

    # FuzzyEn = ln(Phi_m) - ln(Phi_{m+1})
    try:
        phi_m = phi(m)
        phi_m_plus_1 = phi(m + 1)
        
        # Avoid issues with log(0)
        if phi_m == 0.0 or phi_m_plus_1 == 0.0:
            return np.nan

        return np.log(phi_m) - np.log(phi_m_plus_1)
    except Exception as e:
        print(f"Error calculating FuzzyEn: {e}")
        return np.nan

def extract_all_time_domain_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Extracts a set of time-domain features (like ApEn, FuzzyEn, etc.) for a single epoch.
    
    Args:
        epoch_data (np.ndarray): 2D array (n_channels, n_samples) for one epoch.
        sfreq (float): Sampling frequency, required for some features (like Hjorth parameters).

    Returns:
        np.ndarray: A 1D array of extracted features.
    """
    # Import antropy inside the function to keep the dependency local for cleaner main script
    from antropy import hjorth_params, dfa, sample_entropy, num_zerocross
    
    n_channels, n_samples = epoch_data.shape
    features = []

    for i in range(n_channels):
        channel_signal = epoch_data[i, :]
        
        # 1. ApEn and FuzzyEn
        features.append(approximate_entropy(channel_signal))
        features.append(fuzzy_entropy(channel_signal))

        # 2. Hjorth Parameters (Activity, Mobility, Complexity)
        h_params = hjorth_params(channel_signal)
        features.extend([h_params[0], h_params[1], h_params[2]])

        # 3. Fractal Dimension (DFA)
        # Using a fixed default window of N/2 
        # Note: DFA is computationally expensive, but a key feature.
        features.append(dfa(channel_signal, overlap=True, window_size=int(n_samples/2)))

        # 4. Sample Entropy (SampEn)
        # Less sensitive to length than ApEn
        r_sampen = 0.2 * np.std(channel_signal, ddof=1)
        features.append(sample_entropy(channel_signal, m=EMBEDDING_DIM, r=r_sampen)[-1])
        
        # 5. Zero Crossing Count
        features.append(num_zerocross(channel_signal))

    return np.array(features)