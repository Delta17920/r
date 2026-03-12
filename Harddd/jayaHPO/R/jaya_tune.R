#' Hyperparameter Tuning Using the Jaya Algorithm
#'
#' @description
#' Optimizes a user-defined objective function over a mixed-type parameter
#' space containing continuous, integer, and categorical parameters.
#' Internally operates on a normalized [0, 1]^D search space and decodes
#' each candidate into valid hyperparameters before evaluation.
#'
#' @param fn A function that accepts a named list of decoded hyperparameters
#'   and returns a single numeric scalar to be minimized.
#' @param param_space A list of parameter definitions. Each element is a list
#'   with fields:
#'   \describe{
#'     \item{name}{Character. Parameter name.}
#'     \item{type}{Character. One of \code{"continuous"}, \code{"integer"},
#'       or \code{"categorical"}.}
#'     \item{lower}{Numeric. Lower bound (continuous and integer only).}
#'     \item{upper}{Numeric. Upper bound (continuous and integer only).}
#'     \item{scale}{Character. Optional. \code{"log"} for log-scale decoding
#'       (continuous only).}
#'     \item{categories}{Character vector. Category levels (categorical only).}
#'   }
#' @param pop_size Integer. Population size. Default is 10.
#' @param max_iter Integer. Maximum number of iterations. Default is 50.
#' @param seed Integer or NULL. Random seed for reproducibility. Default is NULL.
#'
#' @return A list with components:
#'   \describe{
#'     \item{best_params}{Named list of the best decoded hyperparameter values.}
#'     \item{best_score}{Numeric. The best objective value achieved.}
#'     \item{history}{Numeric vector of best scores per iteration.}
#'   }
#'
#' @examples
#' param_space <- list(
#'   list(name = "x1", type = "continuous", lower = 0, upper = 1),
#'   list(name = "x2", type = "integer",    lower = 1, upper = 10)
#' )
#' result <- jaya_tune(
#'   fn          = function(p) p$x1^2 + (p$x2 - 5)^2,
#'   param_space = param_space,
#'   pop_size    = 5,
#'   max_iter    = 10,
#'   seed        = 42
#' )
#' result$best_params
#' result$best_score
#'
#' @export
jaya_tune <- function(fn, param_space, pop_size = 10, max_iter = 50, seed = NULL) {

  # Input validation
  if (!is.function(fn))
    stop("'fn' must be a function.")
  if (!is.list(param_space) || length(param_space) == 0)
    stop("'param_space' must be a non-empty list.")
  if (!is.numeric(pop_size) || pop_size < 2)
    stop("'pop_size' must be an integer >= 2.")
  if (!is.numeric(max_iter) || max_iter < 1)
    stop("'max_iter' must be a positive integer.")

  valid_types <- c("continuous", "integer", "categorical")
  for (p in param_space) {
    if (!p$type %in% valid_types)
      stop(sprintf("Unknown parameter type '%s' for parameter '%s'.", p$type, p$name))
    if (p$type %in% c("continuous", "integer")) {
      if (is.null(p$lower) || is.null(p$upper))
        stop(sprintf("Parameter '%s' requires 'lower' and 'upper'.", p$name))
      if (p$lower >= p$upper)
        stop(sprintf("'lower' must be less than 'upper' for parameter '%s'.", p$name))
    }
    if (p$type == "categorical") {
      if (is.null(p$categories) || length(p$categories) < 1)
        stop(sprintf("Parameter '%s' requires a non-empty 'categories' vector.", p$name))
    }
  }

  if (!is.null(seed)) set.seed(seed)

  D <- length(param_space)

  # Objective wrapper: decodes internal [0,1]^D vector then calls fn
  internal_obj <- function(x_internal) {
    decoded <- decode_params(x_internal, param_space)
    fn(decoded)
  }

  # Jaya::jaya uses apply() internally on a popSize x n_var matrix.
  # When n_var = 1, that matrix drops to a vector and apply() fails with
  # "dim(X) must have a positive length". We pad to n_var = 2 in that case,
  # passing only the first dimension to the real objective.
  needs_padding <- D == 1L
  if (needs_padding) {
    padded_obj <- function(x_internal) internal_obj(x_internal[1L])
    run_D      <- 2L
  } else {
    padded_obj <- internal_obj
    run_D      <- D
  }

  # Ensure popSize is always strictly greater than n_var as Jaya requires
  safe_pop <- max(as.integer(pop_size), run_D + 1L)

  result <- Jaya::jaya(
    fun     = padded_obj,
    lower   = rep(0, run_D),
    upper   = rep(1, run_D),
    popSize = safe_pop,
    maxiter = as.integer(max_iter),
    n_var   = run_D
  )

  # Extract only the D real dimensions from the best solution
  best_raw    <- as.numeric(result$Best[seq_len(run_D)])[seq_len(D)]
  best_params <- decode_params(best_raw, param_space)
  best_score  <- as.numeric(result$Best[run_D + 1L])
  history     <- result$Iterations

  list(
    best_params = best_params,
    best_score  = best_score,
    history     = history
  )
}


#' @noRd
decode_params <- function(x_internal, param_space) {

  x_internal <- pmin(pmax(x_internal, 0), 1)  # boundary clamp
  decoded    <- vector("list", length(param_space))

  for (i in seq_along(param_space)) {
    p   <- param_space[[i]]
    val <- x_internal[i]

    if (p$type == "continuous") {
      if (!is.null(p$scale) && p$scale == "log") {
        decoded[[i]] <- exp(log(p$lower) + val * (log(p$upper) - log(p$lower)))
      } else {
        decoded[[i]] <- p$lower + val * (p$upper - p$lower)
      }

    } else if (p$type == "integer") {
      decoded[[i]] <- floor(p$lower + val * (p$upper - p$lower + 0.999))
      decoded[[i]] <- max(p$lower, min(decoded[[i]], p$upper))

    } else if (p$type == "categorical") {
      N            <- length(p$categories)
      idx          <- min(floor(val * N) + 1L, N)
      decoded[[i]] <- p$categories[idx]
    }
  }

  stats::setNames(decoded, vapply(param_space, `[[`, character(1), "name"))
}
