test_that("all rcupp operators are reachable through LinkingTo", {
  set.seed(1)
  X <- matrix(rnorm(20 * 4), 20, 4)
  v <- rnorm(4)
  t <- 0.3
  r <- cudaBio:::rcupp_ops_check_cpp(X, v, t)

  expect_equal(r$gemv, drop(X %*% v), tolerance = 1e-12)
  expect_equal(r$gemv_t, drop(crossprod(X, drop(X %*% v))), tolerance = 1e-12)
  expect_equal(r$axpy, 1.5 * v, tolerance = 1e-12)
  expect_equal(r$scal, 2 * v, tolerance = 1e-12)
  expect_equal(r$dot, sum(v * v), tolerance = 1e-12)
  expect_equal(r$nrm2, sqrt(sum(v * v)), tolerance = 1e-12)
  expect_equal(r$soft_threshold, sign(v) * pmax(abs(v) - t, 0),
               tolerance = 1e-12)
  expect_type(r$cuda, "logical")
})

test_that("cuda_pca recovers a dominant rank-1 component", {
  set.seed(2)
  n <- 30L
  p <- 6L
  # A strong rank-1 signal gives a wide eigenvalue gap -> fast convergence.
  X <- matrix(rnorm(n * p, sd = 0.2), n, p) +
    8 * outer(rnorm(n), rnorm(p))
  s <- svd(X)

  fit <- cuda_pca(X, ncomp = 1, max_iter = 1000L, tol = 1e-12)
  expect_s3_class(fit, "cuda_pca")
  expect_equal(dim(fit$loadings), c(p, 1L))
  expect_equal(dim(fit$scores), c(n, 1L))
  # Eigenvector sign is arbitrary; compare the absolute inner product.
  expect_true(abs(sum(fit$loadings[, 1L] * s$v[, 1L])) > 1 - 1e-6)
  expect_equal(as.numeric(fit$values[1L]), s$d[1L]^2, tolerance = 1e-5)
  expect_equal(fit$backend,
               if (rcupp::cuda_available()) "cuda" else "cpu-fallback")
})

test_that("cuda_pca validates inputs", {
  X <- matrix(rnorm(20 * 3), 20, 3)
  expect_error(cuda_pca(matrix("a", 2, 2)), "numeric matrix")
  expect_error(cuda_pca(X, ncomp = 0), "between 1")
  expect_error(cuda_pca(X, ncomp = 4), "between 1")
  expect_error(cuda_pca(X, max_iter = 0), "positive integer")
  expect_error(cuda_pca(X, tol = -1), "positive")
})
