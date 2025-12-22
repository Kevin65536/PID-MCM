To build a synthetic dataset for Partial Information Decomposition (PID) with continuous time series, the most robust approach is to use a **Structural Equation Model (SEM)** combined with **Vector Autoregressive (VAR)** processes.

This design allows you to generate time series where the "Ground Truth" information atoms (Redundancy, Synergy, Unique) are mathematically built into the system via independent latent drivers.

### The Core Concept: Latent Variable Decomposition
The idea is to construct your observable variables ($X_1, X_2, Y$) from hidden, independent information sources (latent variables). By routing these latent sources to specific combinations of $X_1$, $X_2$, and $Y$, you effectively "plant" the information atoms.

*   **Redundancy**: A latent source $R$ is copied to $X_1$, $X_2$, and $Y$.
*   **Unique**: A latent source $U_1$ is copied to $X_1$ and $Y$ (but not $X_2$).
*   **Synergy**: Two latent sources $S_1, S_2$ are copied to $X_1$ and $X_2$ respectively, but $Y$ is computed as their sum (or interaction).

### Step-by-Step Dataset Design

#### 1. Generate Latent Time Series
First, generate 5 independent time series to act as the "carriers" of information. Using AR(1) processes ensures they are continuous and have temporal structure (autocorrelation), which is typical for time-series analysis.

$$ Z_t = \alpha Z_{t-1} + \epsilon_t $$

Where $\epsilon_t \sim \mathcal{N}(0, 1)$ and $\alpha$ (e.g., 0.6) controls the memory of the process.

Generate the following 5 independent latent processes:
*   $R(t)$: Carrier of **Redundant** information.
*   $U_1(t)$: Carrier of **Unique** information for $X_1$.
*   $U_2(t)$: Carrier of **Unique** information for $X_2$.
*   $S_1(t), S_2(t)$: Carriers of **Synergistic** information.

#### 2. Construct Observable Variables ($X_1, X_2, Y$)
Combine the latent variables linearly to create your dataset. You can introduce coefficients ($c$) to tune the strength (signal-to-noise ratio) of each atom.

**Source 1 ($X_1$):**
$$ X_1(t) = c_r R(t) + c_{u1} U_1(t) + c_{s} S_1(t) + \eta_{x1}(t) $$

**Source 2 ($X_2$):**
$$ X_2(t) = c_r R(t) + c_{u2} U_2(t) + c_{s} S_2(t) + \eta_{x2}(t) $$

**Target ($Y$):**
$$ Y(t) = c_r R(t) + c_{u1} U_1(t) + c_{u2} U_2(t) + c_{s} [S_1(t) + S_2(t)] + \eta_{y}(t) $$

*   $\eta$ terms represent observational noise (usually small Gaussian noise) to ensure the system isn't deterministic (which causes infinite mutual information in continuous cases).
*   The term $S_1(t) + S_2(t)$ in $Y$ creates synergy. While $X_1$ knows $S_1$ and $X_2$ knows $S_2$, neither alone can fully explain the sum in $Y$. They must be "joined" (synergy) to reduce the uncertainty of the sum.

#### 3. Tuning the Atoms
By setting the coefficients ($c$), you can create specific test cases:

| Dataset Type | $c_r$ (Red) | $c_{u}$ (Unq) | $c_{s}$ (Syn) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Redundancy** | 1.0 | 0.0 | 0.0 | $X_1, X_2$ both contain the info found in $Y$. |
| **Pure Unique** | 0.0 | 1.0 | 0.0 | $X_1$ has info about $Y$ that $X_2$ doesn't, and vice versa. |
| **Pure Synergy** | 0.0 | 0.0 | 1.0 | $Y$ is the sum of independent components in $X_1, X_2$. |
| **Mixture** | 1.0 | 1.0 | 1.0 | A complex case containing all atoms. |

### Python Implementation Code
Here is a complete script to generate this synthetic dataset.

```python
import numpy as np
import pandas as pd

def generate_ar1(n_samples, alpha=0.6, noise_std=1.0):
    """Generates an AR(1) time series."""
    data = np.zeros(n_samples)
    for t in range(1, n_samples):
        data[t] = alpha * data[t-1] + np.random.normal(0, noise_std)
    return data

def build_pid_dataset(n_samples=5000, cr=1.0, cu=1.0, cs=1.0, noise_lvl=0.1):
    """
    Constructs a PID dataset with tunable Redundancy, Unique, and Synergy atoms.
    
    Args:
        n_samples: Length of time series.
        cr: Coefficient for Redundancy.
        cu: Coefficient for Unique information.
        cs: Coefficient for Synergy.
        noise_lvl: Standard deviation of additive noise on observables.
    """
    # 1. Generate Independent Latent Drivers (Information Atoms)
    # alpha=0.6 gives them temporal structure (making them valid time series)
    L_red = generate_ar1(n_samples)      # Common source
    L_unq1 = generate_ar1(n_samples)     # Unique to X1
    L_unq2 = generate_ar1(n_samples)     # Unique to X2
    L_syn1 = generate_ar1(n_samples)     # Synergistic part 1
    L_syn2 = generate_ar1(n_samples)     # Synergistic part 2

    # 2. Construct Observables via Mixing
    # X1 gets Redundant, Unique1, and Synergy1
    X1 = (cr * L_red) + (cu * L_unq1) + (cs * L_syn1)
    
    # X2 gets Redundant, Unique2, and Synergy2
    X2 = (cr * L_red) + (cu * L_unq2) + (cs * L_syn2)
    
    # Y gets Redundant, Unique1, Unique2, and the Sum of Synergy parts
    # The sum (Syn1 + Syn2) is the classic continuous synergy structure
    Y = (cr * L_red) + (cu * L_unq1) + (cu * L_unq2) + (cs * (L_syn1 + L_syn2))

    # 3. Add observational noise (crucial for continuous entropy calculations)
    X1 += np.random.normal(0, noise_lvl, n_samples)
    X2 += np.random.normal(0, noise_lvl, n_samples)
    Y  += np.random.normal(0, noise_lvl, n_samples)

    return pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# --- Example Usage ---

# Case 1: Redundancy Dominated
df_red = build_pid_dataset(cr=2.0, cu=0.1, cs=0.1)

# Case 2: Synergy Dominated (The "Sum" channel)
df_syn = build_pid_dataset(cr=0.1, cu=0.1, cs=2.0)

# Case 3: Balanced Mixture
df_mix = build_pid_dataset(cr=1.0, cu=1.0, cs=1.0)

print("Dataset Generated. Head of Mixed Case:")
print(df_mix.head())
```

### Why this design works for PID evaluation:
1.  **Clear Ground Truth**: You know exactly where the information came from. If your model detects high unique information in the "Pure Redundancy" dataset, the model is failing.
2.  **Continuity**: The use of Gaussian sums ensures the variables are continuous.
3.  **Stationarity & Dynamics**: The AR(1) process ensures the data looks like real physical time series (e.g., stock prices, sensor readings) rather than white noise, testing your model's ability to handle temporal dependencies (if your PID estimator supports that).
4.  **Linear Gaussian Properties**: This setup produces a **Multivariate Gaussian** distribution. This is the "canonical" benchmark because for this specific class of distributions, exact PID values can often be calculated analytically (depending on the PID definition you subscribe to, such as MMI), allowing for precise quantitative error checking.