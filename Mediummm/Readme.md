# GSoC 2026 Medium Test

## Mixed-Type Encoding Strategy for Jaya

**Applicant:** Pratik Bangerwa  
**Project:** Jaya for Modern Hyperparameter Optimization

## 1. Introduction

The **Jaya optimization algorithm** is a population-based metaheuristic designed for continuous optimization problems. The update rule moves candidate solutions toward the best solution and away from the worst solution:

$$X_{new,i,j} = X_{i,j} + r_1 (X_{best,j} - |X_{i,j}|) - r_2 (X_{worst,j} - |X_{i,j}|)$$

where:

| Symbol | Meaning |
|--------|---------|
| $i$ | Candidate solution index |
| $j$ | Dimension index |
| $r_1, r_2 \sim U(0,1)$ | Random uniform scalars |

While this formulation works well for **continuous optimization**, modern **machine learning hyperparameter tuning** requires handling **mixed parameter types**, including:

- Continuous parameters (e.g., learning rate)
- Integer parameters (e.g., tree depth)
- Categorical parameters (e.g., kernel type)
- Conditional parameters (e.g., model-specific hyperparameters)

Frameworks such as Optuna and Hyperopt support these mixed search spaces through internal encoding strategies. To enable similar functionality for Jaya, we propose an **Encoding-Decoding Layer** that maps Jaya's continuous search space to valid machine learning hyperparameters.

## 2. Internal Continuous Search Space

To ensure numerical stability and fairness across parameters with different ranges, the optimizer maintains an internal normalized search space:

$$X_{internal} \in [0,1]^D$$

where $D$ is the number of hyperparameters.

**Advantages:**

- Prevents parameters with large numeric ranges from dominating the search
- Simplifies boundary handling
- Makes encoding consistent across all parameter types

Before evaluating the objective function, the internal vector is decoded into valid hyperparameters.

## 3. Encoding-Decoding Strategy

### 3.1 Continuous Parameters

Examples: learning rate, regularization strength.

**Linear Scale Decoding:**

$$\text{Value} = \text{Lower} + X_{internal} \cdot (\text{Upper} - \text{Lower})$$

*Example:*

```
eta ∈ [0.01, 0.3]
x_internal = 0.5
→ eta = 0.155
```

**Log-Scale Decoding (Recommended Improvement):**

Many ML hyperparameters perform better when searched on a **logarithmic scale**:

$$\text{Value} = \exp\!\Big(\log(\text{Lower}) + X_{internal} \cdot \big(\log(\text{Upper}) - \log(\text{Lower})\big)\Big)$$

This is strongly recommended for parameters like learning rate or regularization strength that span multiple orders of magnitude.

### 3.2 Integer Parameters

Examples: tree depth, number of estimators, `mtry`.

Jaya treats these internally as continuous values. Decoding formula:

$$\text{Value} = \left\lfloor \text{Lower} + X_{internal} \cdot (\text{Upper} - \text{Lower} + 0.999) \right\rfloor$$

*Example:*

```
max_depth ∈ [3, 10]
x_internal = 0.9
→ max_depth = 10
```

Integer decoding introduces **plateaus in the loss landscape**, but Jaya's distance-based update mechanism can traverse these discrete regions effectively where gradient-based methods fail.

### 3.3 Categorical Parameters

Examples: kernel type, optimizer choice, activation function.

Let:

```r
categories <- c("linear", "poly", "rbf")
N <- 3
```

Decoding rule (1-indexed, consistent with R):

$$\text{Index} = \left\lfloor X_{internal} \times N \right\rfloor + 1$$

*Example:*

```
x_internal = 0.2
Index = floor(0.2 * 3) + 1 = 1
→ "linear"
```

**Safe Indexing** to guard against the edge case $X_{internal} = 1.0$ causing an out-of-bounds index:

```r
index <- min(floor(x_internal * N) + 1, N)
```

### 3.4 Ordinal Parameters (Optional Extension)

Some parameters are **ordered but discrete**, such as:

```r
c("small", "medium", "large")
```

These are encoded as integer levels to preserve the ordering relationship. The formula uses `N - 1` in the denominator (instead of `N` used for unordered categoricals) so that the endpoints `"small"` and `"large"` are each reachable with equal probability, spacing the $N$ levels evenly across $[0, 1]$:

$$\text{Index} = \left\lfloor X_{internal} \cdot (N - 1) \right\rfloor + 1$$

## 4. Conditional / Hierarchical Parameters

Many machine learning models require **conditional hyperparameters**. For example:

```
model = "RandomForest"  →  tune: mtry, max_depth
model = "XGBoost"       →  tune: eta, gamma
```

**Strategy:** The internal vector contains **all possible parameters across all models**:

```
[model_type, rf_mtry, rf_depth, xgb_eta, xgb_gamma]
```

**Decoding procedure:**

1. Decode `model_type` first.
2. Activate only the parameters relevant to that model.
3. Silently ignore all inactive parameters during that evaluation.

Population-based optimizers like Jaya tolerate these **inactive dimensions** well, retaining genetic diversity in dormant parameters until they become relevant under a specific model configuration. The pseudocode in Section 8 demonstrates this with an explicit `active_params` check.

## 5. Boundary Handling

Jaya updates may produce values outside $[0, 1]$. To ensure all decoded results remain feasible, values are clamped before decoding:

```r
x_internal <- pmin(pmax(x_internal, 0), 1)
```

This step is applied **before** any type-specific decoding, guaranteeing all downstream formulas receive valid inputs.

## 6. Caching Using Decoded Hashing

Due to integer rounding and categorical binning, multiple distinct continuous vectors can decode to the **same hyperparameter configuration**. For example:

```
x1 = [0.81]  →  max_depth = 9
x2 = [0.83]  →  max_depth = 9
```

To avoid redundant, computationally expensive model training runs, decoded configurations are hashed:

```r
key <- rlang::hash(decoded_params)
```

If the key already exists in the cache, the stored score is returned immediately without re-training.

## 7. Parallel Evaluation

Jaya evaluates an entire **population of candidate solutions per iteration**, making it naturally parallelizable. Each candidate is independent, so all can be evaluated concurrently:

```r
future.apply::future_lapply(population, evaluate_candidate)
```

This enables full multi-core CPU utilization and significantly faster wall-clock search times, especially when individual model training is expensive.

## 8. Implementation Pseudocode

```r
decode_hyperparameters <- function(x_internal, param_space, active_params = NULL) {
  decoded_params <- list()
  x_internal <- pmin(pmax(x_internal, 0), 1)  # Boundary clamp

  for (i in seq_along(param_space)) {
    p   <- param_space[[i]]
    val <- x_internal[i]

    # Skip inactive parameters (for conditional/hierarchical spaces)
    if (!is.null(active_params) && !(p$name %in% active_params)) next

    if (p$type == "continuous") {
      if (!is.null(p$scale) && p$scale == "log") {
        decoded_params[[p$name]] <- exp(
          log(p$lower) + val * (log(p$upper) - log(p$lower))
        )
      } else {
        decoded_params[[p$name]] <- p$lower + val * (p$upper - p$lower)
      }

    } else if (p$type == "integer") {
      decoded_params[[p$name]] <- floor(
        p$lower + val * (p$upper - p$lower + 0.999)
      )

    } else if (p$type == "categorical") {
      N   <- length(p$categories)
      idx <- min(floor(val * N) + 1, N)
      decoded_params[[p$name]] <- p$categories[idx]
    }
  }

  return(decoded_params)
}
```

## 9. Proof-of-Concept Test

Define the mixed-type search space:

```r
param_space <- list(
  list(name = "eta",       type = "continuous",  lower = 0.01, upper = 0.3, scale = "log"),
  list(name = "max_depth", type = "integer",     lower = 3,    upper = 10),
  list(name = "kernel",    type = "categorical", categories = c("linear", "poly", "rbf"))
)
```

Simulated internal Jaya vector:

```
[0.5, 0.9, 0.2]
```

Decoded result:

```
eta       = 0.0548   ← log-scale midpoint of [0.01, 0.3]
max_depth = 10       ← correctly scaled and rounded integer
kernel    = "linear" ← first category (R index 1)
```

**Verification of `eta`:**  
$\exp(\log(0.01) + 0.5 \times (\log(0.3) - \log(0.01))) = \exp(-4.605 + 0.5 \times 3.401) \approx 0.0548$ ✓

## 10. Decoder Validation Experiment

To verify robustness, 1000 random candidate vectors are simulated to confirm the decoder never crashes or violates parameter boundaries:

```r
set.seed(42)

results <- replicate(1000, {
  x <- runif(3)
  decode_hyperparameters(x, param_space)
}, simplify = FALSE)

# Verify boundary conditions
eta_vals   <- sapply(results, `[[`, "eta")
depth_vals <- sapply(results, `[[`, "max_depth")
kern_vals  <- sapply(results, `[[`, "kernel")

stopifnot(all(eta_vals   >= 0.01 & eta_vals   <= 0.3))
stopifnot(all(depth_vals >= 3    & depth_vals <= 10))
stopifnot(all(kern_vals  %in% c("linear", "poly", "rbf")))
```

Expected conditions strictly maintained across all 1000 draws:

| Parameter | Valid Range |
|-----------|-------------|
| `eta` | $[0.01,\ 0.3]$ |
| `max_depth` | $\{3, 4, \ldots, 10\}$ |
| `kernel` | $\{\texttt{linear},\ \texttt{poly},\ \texttt{rbf}\}$ |

## 11. Architecture Overview

```
Jaya Optimizer
      │
      ▼
Continuous Vector [0,1]^D
      │
      ▼
Boundary Clamp  ←─── pmin / pmax
      │
      ▼
Encoding-Decoding Layer
  ├── Continuous  (linear / log scale)
  ├── Integer     (floor rounding)
  ├── Categorical (binned indexing)
  └── Conditional (active_params filter)
      │
      ▼
Valid Hyperparameters
      │
      ▼
Cache Lookup ──── hit  → return stored score
      │ miss
      ▼
Model Training
      │
      ▼
Performance Score → Cache Store
      │
      ▼
Jaya Update Rule
```

## 12. Computational Complexity

The Jaya update step has complexity:

$$O(P \times D)$$

where $P$ is the population size and $D$ is the number of parameters.

The **dominant cost in hyperparameter tuning is model training**, not the optimizer itself. The decoding layer adds $O(D)$ overhead per candidate, which is entirely negligible relative to a single model fit.

## 13. Reproducibility

To ensure reproducible results, a random seed must be set before any optimization run:

```r
set.seed(42)
```

This guarantees consistent candidate generation during testing, validation, and across parallel worker processes.

## 14. Advantages

| Feature | Benefit |
|---------|---------|
| Normalized $[0,1]^D$ internal space | Prevents scale dominance across parameters |
| Mixed-type decoding | Supports Continuous, Log, Integer, Categorical, Ordinal |
| Conditional parameter support | Handles hierarchical model-specific search spaces |
| Boundary clamping | Guarantees feasibility after every Jaya update |
| Hash-based caching | Eliminates redundant model evaluations |
| Population-level parallelism | Enables full multi-core utilization |

## 15. Conclusion

This encoding-decoding framework enables the **Jaya algorithm to function as a modern hyperparameter optimization engine** capable of handling the mixed parameter spaces required by contemporary machine learning workflows.

By combining a normalized internal search space, mixed-type decoding, robust boundary control, conditional parameter handling, hash-based caching, and a validation experiment, Jaya can be integrated effectively into production ML pipelines while fully preserving its derivative-free, parameter-less optimization properties.
