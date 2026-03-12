test_that("jaya_tune returns correct structure", {
  param_space <- list(
    list(name = "x1", type = "continuous", lower = 0, upper = 1)
  )
  result <- jaya_tune(
    fn          = function(p) p$x1^2,
    param_space = param_space,
    pop_size    = 5,
    max_iter    = 5,
    seed        = 42
  )
  expect_type(result, "list")
  expect_named(result, c("best_params", "best_score", "history"))
  expect_true(is.numeric(result$best_score))
  expect_length(result$history, 5)
})

test_that("integer parameters are decoded to integers", {
  param_space <- list(
    list(name = "depth", type = "integer", lower = 1, upper = 10)
  )
  result <- jaya_tune(
    fn          = function(p) abs(p$depth - 5),
    param_space = param_space,
    pop_size    = 5,
    max_iter    = 5,
    seed        = 42
  )
  expect_true(result$best_params$depth == floor(result$best_params$depth))
  expect_true(result$best_params$depth >= 1)
  expect_true(result$best_params$depth <= 10)
})

test_that("categorical parameters return valid category", {
  param_space <- list(
    list(name = "kernel", type = "categorical",
         categories = c("linear", "poly", "rbf"))
  )
  result <- jaya_tune(
    fn          = function(p) as.numeric(p$kernel != "rbf"),
    param_space = param_space,
    pop_size    = 5,
    max_iter    = 5,
    seed        = 42
  )
  expect_true(result$best_params$kernel %in% c("linear", "poly", "rbf"))
})

test_that("log-scale continuous parameter stays in bounds", {
  param_space <- list(
    list(name = "lr", type = "continuous",
         lower = 0.0001, upper = 0.1, scale = "log")
  )
  result <- jaya_tune(
    fn          = function(p) abs(p$lr - 0.01),
    param_space = param_space,
    pop_size    = 5,
    max_iter    = 5,
    seed        = 42
  )
  expect_true(result$best_params$lr >= 0.0001)
  expect_true(result$best_params$lr <= 0.1)
})

test_that("jaya_tune errors on bad inputs", {
  param_space <- list(
    list(name = "x", type = "continuous", lower = 0, upper = 1)
  )
  expect_error(jaya_tune(fn = "not_a_function", param_space = param_space))
  expect_error(jaya_tune(fn = function(p) 0, param_space = list()))
  expect_error(jaya_tune(fn = function(p) 0, param_space = param_space,
                         pop_size = 1))
})

test_that("decode_params boundary clamp works", {
  param_space <- list(
    list(name = "x", type = "continuous", lower = 0, upper = 1)
  )
  result_low  <- jayaHPO:::decode_params(c(-0.5), param_space)
  result_high <- jayaHPO:::decode_params(c(1.5),  param_space)
  expect_true(result_low$x  >= 0)
  expect_true(result_high$x <= 1)
})
