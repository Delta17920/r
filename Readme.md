# GSoC 2026 - Jaya for Modern Hyperparameter Optimization

**Applicant:** Pratik Bangerwa  
**Email:** pratik161106@gmail.com  
**Project:** Jaya for Modern Hyperparameter Optimization  
**Organization:** R Project for Statistical Computing

## About the Project

The [Jaya R package](https://cran.r-project.org/package=Jaya) implements a gradient-free, population-based optimization algorithm known for its parameter-free design. This GSoC project proposes to extend Jaya into a modern hyperparameter optimization engine for R, with native support for mixed parameter types and seamless integration with ML ecosystems like mlr3, tidymodels, and caret.

## Repository Structure

```
├── Easyyy/        ← Easy Test:   Neural Network HPO using Jaya
├── Mediummm/      ← Medium Test: Mixed-Type Encoding Strategy for Jaya
└── Harddd/        ← Hard Test:   jayaHPO R package with jaya_tune()
```

## Tests

### Easy Test - [`Easyyy/`](https://github.com/delta17920/r/tree/main/Easyyy)

Optimizes a Single-Hidden-Layer Neural Network (`nnet`) on the `Pima` diabetes dataset using the Jaya algorithm. Simultaneously tunes two hyperparameters:

| Parameter | Type | Range |
|-----------|------|-------|
| `size` | Integer | 1 to 20 hidden neurons |
| `decay` | Continuous (log-scale) | 0.0001 to 0.1 |

The Brier Score (MSE of predicted probabilities) is used as the objective instead of flat classification accuracy, providing a smooth continuous signal for the optimizer to descend.

**Files:**
- [`easy_test.Rmd`](https://github.com/delta17920/r/blob/main/Easyyy/Jaya_Easy_Test.Rmd) - full reproducible RMarkdown source
- [`README.md`](https://github.com/delta17920/r/blob/main/Easyyy/Readme.md) - knitted github_document output with results and convergence plot

### Medium Test - [`Mediummm/`](https://github.com/delta17920/r/tree/main/Mediummm)

Proposes and justifies a complete encoding-decoding strategy enabling Jaya to operate over mixed-type hyperparameter spaces. Covers continuous (linear and log-scale), integer, categorical, ordinal, and conditional parameters. Includes implementation pseudocode, a proof-of-concept test, and a 1000-sample validation experiment.

**Files:**
- [`Readme.md`](https://github.com/delta17920/r/blob/main/Mediummm/Readme.md) - full proposal document

### Hard Test - [`Harddd/`](https://github.com/delta17920/r/tree/main/Harddd)

A minimal but fully functional R package implementing `jaya_tune()`, a hyperparameter optimization function built on the Jaya algorithm with native support for continuous, integer, and categorical parameters via an encoding-decoding layer.

**Highlights:**
- Normalized `[0,1]^D` internal search space
- Floor-based integer decoding with uniform bucket widths
- Log-scale decoding for parameters spanning multiple orders of magnitude
- Input validation with informative error messages
- 15 unit tests via `testthat` covering all parameter types and edge cases
- Verified clean on win-builder (R-devel and R-release) with 0 errors and 0 warnings

**Files:**
- [`jayaHPO/`](https://github.com/delta17920/r/tree/main/Harddd/jayaHPO) - complete R package source
- [`jayaHPO/R/jaya_tune.R`](https://github.com/delta17920/r/blob/main/Harddd/jayaHPO/R/jaya_tune.R) - core implementation
- [`jayaHPO/tests/testthat/test-jaya_tune.R`](https://github.com/delta17920/r/blob/main/Harddd/jayaHPO/tests/testthat/test-jaya_tune.R) - unit tests
- [`jayaHPO/proof/`](https://github.com/delta17920/r/tree/main/Harddd/jayaHPO/proof) - win-builder and test screenshots

## Key Design Principle

All three tests share a consistent internal architecture. Jaya always operates on a normalized `[0,1]^D` search space. Decoding to valid hyperparameters happens via a type-specific layer before each evaluation:

```
Jaya Optimizer → Continuous Vector [0,1]^D → Encoding-Decoding Layer → Valid Hyperparameters → Objective Function
```

This design prevents scale dominance, simplifies boundary handling, and keeps Jaya's core update rule untouched.
