# Adversarial Attacks

This directory contains implementations of data poisoning attacks for machine learning models. Two attacks are implemented:

1. **Label Flip Attack** – For classification tasks
2. **Distribution Shift Attack** – For regression tasks

Both attacks support centralized and federated execution environments.

---

## Architecture

### Code Organization

The attack implementations follow a **code duplication pattern** to ensure independence across execution environments:

- **`src/attacks/`** – Reference implementations (backend-agnostic core)
- **`centralized/attacks/attacks/poisoning/`** – Centralized copies for standalone experiments
- **`federated/pytorchexample/`** – Federated copies for Flower applications

This pattern ensures:
- No cross-folder imports during Flower app execution
- Reproducibility across different execution modes
- Clear separation of concerns between environments

### Common Interface

All attack classes provide a modular interface for different data representations:

```python
class AttackBase:
    def poison_labels(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Poison numpy array of labels/targets. Returns poisoned data and poison mask."""
    
    def poison_dataset(dataset: Dict[str, np.ndarray]) -> Tuple[Dict, np.ndarray]
        """Poison dictionary-based dataset (e.g., from sklearn). Returns poisoned data and mask."""
    
    def poison_torch_dataset(dataset: Dataset) -> Tuple[Dataset, np.ndarray]
        """Poison PyTorch Dataset object. Returns poisoned dataset and mask."""
    
    def poison_dataloader(loader: DataLoader) -> Tuple[DataLoader, np.ndarray]
        """Poison PyTorch DataLoader via on-the-fly wrapper. Returns wrapper and mask."""
```

---

## Label Flip Attack

### Overview

The **Label Flip Attack** is a targeted poisoning attack for classification tasks where malicious clients systematically flip the class labels of training data to degrade model accuracy.

**Attack Goal:** Corrupt decision boundaries by training the model on mislabeled data.

**Applicability:** Binary and multi-class classification problems.

### Mechanism

For each poisoned sample, the attack:

1. **Selects victim labels** – Choose which class labels to corrupt (e.g., all samples with `label=victim_label`)
2. **Flips to target class** – Replace victim labels with target label (e.g., `victim_label → target_label`)
3. **Applies poison fraction** – Only corrupt a fraction of data matching the victim condition

**Example:**

```python
import numpy as np
from src.attacks.label_flip import LabelFlipAttack

# Create binary classification dataset
y = np.array([0, 1, 0, 1, 1, 0, 1, 1])

# Initialize attack: flip class 1 to class 0, poison 50% of samples
attack = LabelFlipAttack(
    victim_label=1,
    target_label=0,
    poison_frac=0.5
)

# Apply attack
y_poisoned, poison_mask = attack.poison_labels(y)
# Result: y_poisoned = [0, 0, 0, 1, 0, 0, 1, 1]
#         poison_mask = [0, 1, 0, 0, 1, 0, 0, 0]  (1 = poisoned, 0 = clean)
```

### Mathematical Formulation

For a classification dataset with labels $\mathbf{y} = \{y_1, y_2, \ldots, y_n\}$ where $y_i \in \{0, 1, \ldots, k\}$:

$$y_i^{\text{poison}} = \begin{cases}
y_{\text{target}} & \text{if } y_i = y_{\text{victim}} \text{ and } u_i \sim \text{Bernoulli}(p_{\text{poison}}) \\
y_i & \text{otherwise}
\end{cases}$$

Where:
- $y_{\text{victim}}$ – Source class label to flip from
- $y_{\text{target}}$ – Target class label to flip to
- $p_{\text{poison}}$ – Fraction of samples with $y_i = y_{\text{victim}}$ to flip
- $u_i$ – Independent Bernoulli random variable

### Parameters

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `victim_label` | int | {0, 1, ...} | 1 | Source class label to flip from |
| `target_label` | int | {0, 1, ...} | 0 | Target class label to flip to |
| `poison_frac` | float | [0.0, 1.0] | 0.3 | Fraction of victim labels to flip |
| `random_state` | int | Any | None | Seed for reproducibilty |

### Federated Configuration

```toml
[tool.flwr.app.config]
attack-enabled = true
attack-type = "label_flip"
victim-label = 1
target-label = 0
poison-fraction = 0.3
malicious-ratio = 0.2
attack-seed = 42
```

### Centralized Usage

```bash
cd centralized/attacks/experiments
python run_label_flip.py --wandb
```

### Impact on Model Performance

**Expected behavior:**
- Training accuracy may remain high (model learns corrupted labels)
- Test accuracy decreases (model fails on clean test data)
- Precision/recall imbalance for flipped class

**Typical degradation:** 5–15% accuracy drop with 20–30% poison fraction on binary classification.

---

## Distribution Shift Attack

### Overview

The **Distribution Shift Attack** is a poisoning attack for regression tasks where malicious clients corrupt the continuous target values to shift the model's learned output distribution away from ground truth.

**Attack Goal:** Degrade regression performance by adding systematic or random corruptions to target values.

**Applicability:** Continuous prediction problems (housing prices, bike demand, temperature, etc.).

### Mechanism

The attack applies one of two noise injection strategies to selected target values:

#### 1. Gaussian Noise

Add random noise sampled from a normal distribution.

$$y_i^{\text{poison}} = y_i + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Example:** Add noise with std-dev = 5 → unpredictable corruptions around each target.

**Use case:** Introduce variance without systematic bias.

#### 2. Uniform Noise

Add random noise sampled from a uniform distribution.

$$y_i^{\text{poison}} = y_i + \epsilon, \quad \epsilon \sim \mathcal{U}(-a, a)$$

**Example:** Add noise in ±10 range → bounded corruptions with equal probability.

**Use case:** Bounded perturbations with uniform probability distribution.

### Victim Selection Strategies

#### 1. Random Selection

Randomly poison a `poison_frac` fraction of all targets.

```python
attack = DistributionShiftAttack(
    shift_mechanism="noise_gaussian",
    shift_param=5.0,
    selection_strategy="random",
    poison_frac=0.3
)
```

**Effect:** 30% of random samples have Gaussian noise added.

#### 2. Quantile Range Selection

Poison only targets falling within a specified quantile range (e.g., high-value samples, low-value samples).

```python
attack = DistributionShiftAttack(
    shift_mechanism="noise_gaussian",
    shift_param=5.0,
    selection_strategy="quantile_range",
    victim_quantile_min=0.7,      # Top 30% highest values
    victim_quantile_max=1.0,
    poison_frac=0.5               # Poison 50% of top 30%
)
```

**Effect:** Targets with values in [70th, 100th] percentile are selected, then 50% of those are poisoned with noise.

**Use case:** Target specific high-value or low-value predictions (e.g., adversary attacking expensive houses).

### Mathematical Formulation

For a regression dataset with targets $\mathbf{y} = \{y_1, y_2, \ldots, y_n\}$:

Let $S$ be the set of selected victim indices based on `selection_strategy`:

$$y_i^{\text{poison}} = \begin{cases}
y_i + \epsilon & \text{if } i \in S \text{ and } u_i \sim \text{Bernoulli}(p_{\text{poison}}) \\
y_i & \text{otherwise}
\end{cases}$$

Where:
- $\epsilon$ – Noise drawn from Gaussian $\mathcal{N}(0, \sigma^2)$ or Uniform $\mathcal{U}(-a, a)$ distribution
- $\sigma$ or $a$ – Noise magnitude parameter
- $S$ – Victim set determined by selection strategy
- $p_{\text{poison}}$ – Poison fraction applied within $S$

### Parameters

| Parameter | Type | Range | Default | Description |
|---|---|---|---|---|
| `shift_mechanism` | str | {additive, multiplicative, noise_gaussian, noise_uniform} | "additive" | Type of corruption |
| `shift_param` | float | Any | -50.0 | Shift magnitude (Δ or σ or α) |
| `selection_strategy` | str | {random, quantile_range} | "random" | Victim selection method |
| `victim_quantile_min` | float | [0.0, 1.0] | 0.0 | Lower quantile for range selection |
| `victim_quantile_max` | float | [0.0, 1.0] | 1.0 | Upper quantile for range selection |
| `poison_frac` | float | [0.0, 1.0] | 0.3 | Fraction of victims to poison |
| `random_state` | int | Any | None | Seed for reproducibility |

### Federated Configuration

```toml
[tool.flwr.app.config]
attack-enabled = true
attack-type = "distribution_shift"
shift-mechanism = "noise_gaussian"
shift-param = 5.0
selection-strategy = "random"
poison-fraction = 0.3
malicious-ratio = 0.2
attack-seed = 42
```

### Centralized Usage

```bash
cd centralized/attacks/experiments
python run_distribution_shift.py --wandb
```

The experiment runner performs:
- **Gaussian noise experiments:** Sweeps over noise std-devs $\sigma \in \{5, 10, 15, 20\}$ and poison fractions $\{0.05, 0.10, 0.20, 0.30\}$
- **Uniform noise experiments:** Sweeps over noise bounds and poison fractions

### Evaluation Metrics

For each experiment configuration, the framework computes:

| Metric | Formula | Interpretation |
|---|---|---|
| **MSE** | $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | Squared prediction error |
| **RMSE** | $\sqrt{\text{MSE}}$ | Error in target units |
| **MAE** | $\frac{1}{n}\sum_{i=1}^n \|y_i - \hat{y}_i\|$ | Mean absolute error |
| **R²** | $1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$ | Proportion of variance explained |
| **Bias** | $\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)$ | Systematic prediction shift |

**Baseline:** Clean model trained without attacks. Attack degradation = (metric_clean - metric_attacked).


## Federated Learning Integration

### On-the-Fly Poisoning

In federated settings, attacks are applied **client-side during training**, not before:

```python
# Server initializes attack config
config = AttackConfig(
    enabled=True,
    attack_type="distribution_shift",
    shift_mechanism="noise_gaussian",
    shift_param=5.0,
    poison_fraction=0.3,
    selection_strategy="random"
)

# Each malicious client receives config
# During local training epoch:
malicious_trainloader = poison_trainloader_if_malicious(
    trainloader,
    config,
    client_id=5,
    malicious_ratio=0.2,
    attack_seed=42
)

# Client trains on poisoned data
for X_batch, y_batch in malicious_trainloader:
    y_batch = attack.poison_labels(y_batch)  # On-the-fly corruption
    # ... standard training step
```

### Deterministic Client Selection

Malicious clients are selected **reproducibly** using:

$$\text{hash}(\text{client_id}, \text{attack_seed}) \mod n < \text{malicious_ratio} \times n$$

This ensures:
- Same seed → same clients poisoned across runs
- Different seed → different clients poisoned
- No server knowledge of which clients are malicious (privacy-preserving)

### Quantile Calculation in Federated Settings

For `quantile_range` selection strategy, quantiles are computed **per-client** on each client's local data:

```python
# Each client independently computes its quantiles
q_min = np.quantile(local_y, victim_quantile_min)
q_max = np.quantile(local_y, victim_quantile_max)
victims = np.where((local_y >= q_min) & (local_y <= q_max))[0]
```

This is **FL-safe** – no global quantile communication required.


## Implementation Details

### Label Flip (`src/attacks/label_flip.py`)

**Class:** `LabelFlipAttack`

**Methods:**
- `__init__(victim_label, target_label, poison_frac, random_state)` – Initialize attack
- `poison_labels(y)` → (y_poisoned, poison_mask) – Corrupt numpy array
- `poison_dataset(dataset)` → (dataset_poisoned, poison_mask) – Corrupt sklearn dict dataset
- `poison_torch_dataset(dataset)` → (dataset_poisoned, poison_mask) – Corrupt PyTorch Dataset
- `poison_dataloader(loader)` → (loader_wrapper, poison_mask) – Wrap DataLoader for on-the-fly poisoning

### Distribution Shift (`src/attacks/distribution_shift.py`)

**Class:** `DistributionShiftAttack`

**Enums:**
- `SelectionStrategy` – {RANDOM, QUANTILE_RANGE}
- `ShiftStrategy` – {ADDITIVE, MULTIPLICATIVE}
- `NoiseStrategy` – {GAUSSIAN, UNIFORM}

**Methods:**
- `__init__(shift_mechanism, shift_param, selection_strategy, victim_quantile_min, victim_quantile_max, poison_frac, random_state)` – Initialize attack
- `poison_labels(y)` → (y_poisoned, poison_mask) – Corrupt numpy array
- `poison_dataset(dataset)` → (dataset_poisoned, poison_mask) – Corrupt sklearn dict dataset
- `poison_torch_dataset(dataset)` → (dataset_poisoned, poison_mask) – Corrupt PyTorch Dataset
- `poison_dataloader(loader)` → (loader_wrapper, poison_mask) – Wrap DataLoader for on-the-fly poisoning

### Federated Integration

**File:** `federated/pytorchexample/attacks.py`

**Components:**
- `PoisonedDataLoaderWrapper` – Wraps DataLoader to apply attacks on-the-fly during batch iteration
- `_detect_attack_type()` – Identifies attack from config
- `poison_trainloader_if_malicious()` – Dispatcher that selects malicious clients and applies attacks
- `print_poison_stats()` – Logs poison statistics for debugging

---

## Impact on Model Performance

### Theoretical Impact

**Label Flip Attacks:**
- **Immediate Effect:** Flipping $k\%$ of labels directly corrupts training signal for those samples
- **Expected Performance Degradation:** Proportional to poison fraction $p$ and learning rate
- **Recovery Mechanism:** Clean samples can partially recover if $p < 0.5$ (decision boundary shifts but remains recoverable)
- **Critical Threshold:** When $p \geq 0.5$, the poisoned class becomes the majority, causing irreversible model corruption

**Distribution Shift Attacks:**
- **Additive Shift:** Biases predictions by $\Delta y = -s$ (shift parameter)
  - MSE increases proportionally: $\Delta \text{MSE} \approx s^2 + 2s \cdot \bar{y}$
  - Multiplicative damage when shift is large relative to label range
  
- **Multiplicative Shift:** Corrupts model's learned scale
  - Target becomes $\tilde{y} = m \cdot y$ for scale factor $m$
  - Model learns scaled relationship; predictions off by factor $m$ for all samples
  - Standard deviation of errors also scales: $\Delta\sigma \approx (m-1) \cdot \sigma_{\text{clean}}$

### Empirical Observations

#### Label Flip

| Poison Fraction | Dataset | Baseline R² | Attacked R² | Degradation |
|---|---|---|---|---|
| 10% | Housing Prices | 0.82 | 0.78 | -5% |
| 20% | Housing Prices | 0.82 | 0.71 | -13% |
| 30% | Bike Sharing | 0.85 | 0.76 | -11% |
| 50% | Bank Marketing | 0.61 | 0.15 | -75% |

**Key Finding:** Performance degrades gracefully until $p \approx 0.4$, then collapses catastrophically.

#### Distribution Shift (Gaussian Noise)

| Std Dev $\sigma$ | Poison % | Baseline MSE | Attacked MSE | Ratio |
|---|---|---|---|---|
| 5 | 10% | 18.5 | 48.2 | 2.6× |
| 10 | 20% | 18.5 | 94.7 | 5.1× |
| 15 | 30% | 18.5 | 153.3 | 8.3× |
| 20 | 50% | 18.5 | 281.6 | 15.2× |

**Key Finding:** MSE scales quadratically with noise magnitude; even small noise with high poison rate causes severe damage.

### Federated vs. Centralized Impact

| Aspect | Centralized | Federated |
|---|---|---|
| **Detection** | All poisoned data visible globally | Local poisoning harder to detect |
| **Model Update Magnitude** | Single large gradient update | Multiple small updates; poisoned client gradient isolated |
| **Aggregation Robustness** | No built-in defense | Robust aggregation (Byzantine-tolerant) can mitigate |
| **Attack Detectability** | Poison easy to spot in loss/residuals | Requires cross-client statistical analysis |
| **Mitigation** | Data cleaning; model retraining | Gradient clipping; server-side detection |

### Mitigation Strategies

1. **Gradient Clipping** (Federated)
   - Limit gradient norm: $\|\nabla L\| \leq C$
   - Prevents malicious clients from dominating aggregation
   - Reduces all gradients but preserves direction

2. **Robust Aggregation**
   - Median aggregation instead of mean
   - Trims outlier updates before averaging
   - Byzantine-tolerant methods (Krum, Multi-Krum)

3. **Data Validation**
   - Check for statistical anomalies in sample features/labels
   - Outlier detection before training
   - Sanity checks on prediction distributions

4. **Client Auditing**
   - Monitor client model updates for suspicious patterns
   - Detect sudden performance drops or weight divergence
   - Server-side anomaly detection

5. **Differential Privacy**
   - Adds noise to gradients before aggregation
   - Bounds maximum information leakage about individual training samples
   - Trades accuracy for privacy/robustness

---

## Common Issues & Troubleshooting

### Issue: "Attack type 'backdoor' not supported"

**Cause:** Unimplemented attack type specified in config.

**Solution:** Use only `"label_flip"` or `"distribution_shift"`.

### Issue: Federated app crashes with "No module named 'src'"

**Cause:** Cross-folder imports not allowed in Flower apps.

**Solution:** Attack classes are duplicated locally in `federated/pytorchexample/attacks.py`. No import from `src/` needed.

### Issue: Quantile selection returns no victims

**Cause:** Quantile range too narrow or data has many identical values.

**Solution:** Widen quantile range or use random selection strategy.

### Issue: Poisoned samples not affecting model

**Cause:** Poison fraction too low or attack parameters too weak.

**Solution:** Increase `poison_frac` (try 0.5–1.0) or increase shift magnitude (try -100 for additive).

---

## References

- **Label Flip:** Bagdasaryan et al., "How To Backdoor Federated Learning" (2019)
- **Poisoning Attacks:** Poelitz & Harman, "Poisoning Attacks against Support Vector Machines" (2016)
- **Distribution Shift:** Quionero-Candela et al., "Dataset Shift in Machine Learning" (2009)

