GSoC 2026 Medium Test: Mixed-Type Encoding Strategy for Jaya

Applicant: Pratik Bangerwa

Project: Jaya for Modern Hyperparameter Optimization

1. The Core Architectural Challenge

The fundamental Jaya optimization algorithm operates strictly within a continuous numerical domain, updating candidate solutions using spatial distance vectors:

$$X_{new, i, j} = X_{i, j} + r_1(X_{best, j} - |X_{i, j}|) - r_2(X_{worst, j} - |X_{i, j}|)$$

However, modern Machine Learning workflows (e.g., mlr3, tidymodels) demand mixed-type search spaces encompassing Continuous, Integer, Categorical, and Conditional parameters. To position Jaya as a first-class HPO engine, we must implement an invisible Encoding-Decoding Layer that bridges Jaya's continuous mathematics with discrete ML requirements.

2. Mathematical Encoding-Decoding Strategy

To prevent parameters with massive numerical ranges from dominating the distance calculations, the tuning engine maintains an internal continuous search space scaled strictly to $X_{internal} \in [0, 1]^D$.

The decoder translates these normalized spatial dimensions into valid ML hyperparameters immediately prior to objective evaluation.

A. Continuous Parameters (e.g., L2 Regularization, Learning Rate)

Encoding: Normalization to $[0, 1]$.

Decoding: Denormalization using linear scaling.

Formula: 

$$Value = Lower + X_{internal} \times (Upper - Lower)$$

Justification: Preserves Jaya's gradient-free continuous search dynamics without alteration.

B. Integer Parameters (e.g., Tree Depth, mtry)

Encoding: Treated internally as continuous variables in $[0, 1]$.

Decoding: Denormalized to the target range and strictly rounded to the nearest whole number.

Formula: 

$$Value = \lfloor Lower + X_{internal} \times (Upper - Lower + 0.999) \rfloor$$

Justification: Rounding creates "plateaus" in the loss landscape. Gradient-based optimizers fail here (the derivative of a plateau is zero), but Jaya's distance-based search gracefully steps across these discrete plateaus to find integer optimums.

C. Categorical Parameters (e.g., Kernel Type: ['linear', 'poly', 'rbf'])

Encoding: Categorical options are mapped to a 1D array of length $N$. Jaya searches a continuous dimension $X_{internal} \in [0, 1)$.

Decoding: The continuous value is multiplied by $N$ and floored to derive the 1-based R array index.

Formula: 

$$Index = \lfloor X_{internal} \times N \rfloor + 1$$

Justification: Transforms qualitative string choices into a quantitative spatial dimension. By sorting categories logically (e.g., ordering kernels by computational complexity), we establish a pseudo-gradient that Jaya can mathematically navigate.

D. Conditional / Hierarchical Parameters

The Problem: Modern HPO requires conditional logic (e.g., if the optimizer selects "Random Forest", tune mtry; if it selects "XGBoost", tune eta).

Decoding Strategy: The continuous vector $X$ holds hidden dimensions for all possible parameters across all models. However, the decoder evaluates the top-level categorical parameter first.

Justification: If "Random Forest" is selected, the eta dimension is ignored during ML evaluation. While this creates temporary "dead dimensions" in the search space, Jaya's population-based nature allows it to retain genetic diversity in those dormant parameters until an XGBoost configuration is evaluated.

3. Implementation Pseudocode

decode_hyperparameters <- function(x_internal, param_space) {
  # x_internal: A numeric vector from Jaya bounded in [0, 1]
  decoded_params <- list()
  
  for (i in seq_along(param_space)) {
    p <- param_space[[i]]
    val <- x_internal[i]
    
    if (p$type == "continuous") {
      decoded_params[[p$name]] <- p$lower + val * (p$upper - p$lower)
      
    } else if (p$type == "integer") {
      decoded_params[[p$name]] <- floor(p$lower + val * (p$upper - p$lower + 0.999))
      
    } else if (p$type == "categorical") {
      idx <- floor(val * length(p$categories)) + 1
      decoded_params[[p$name]] <- p$categories[idx]
    }
  }
  
  return(decoded_params)
}


4. Efficiency via Decoded Hashing (Caching)

Because Integer and Categorical decoders act as step-functions, multiple different continuous internal vectors will decode to the exact same hyperparameter configuration.

Proposed Mitigation: To prevent re-training the ML model on identical settings, the tuner will implement a hashing mechanism (e.g., rlang::hash()) on the decoded parameter list. Before invoking the computationally expensive training step, the tuner checks a lookup table. If the configuration exists, it instantly returns the cached score, drastically reducing tuning overhead.