#' Principal components by GPU power iteration (rcupp framework example)
#'
#' Computes the leading eigenvectors of \code{crossprod(x)} by power iteration,
#' calling the fixed GPU operators exposed by \pkg{rcupp} through
#' \code{LinkingTo: rcupp} and \code{R_GetCCallable()}. This package ships no
#' CUDA code and no CUDA build logic of its own; it works identically whether
#' \pkg{rcupp} was built with a CUDA backend or its CPU fallback.
#'
#' @param x Numeric matrix \code{n x p}.
#' @param ncomp Number of leading components (between 1 and \code{ncol(x)}).
#' @param max_iter Maximum power-iteration steps per component.
#' @param tol Convergence tolerance on the change of the unit eigenvector.
#' @return A list of class \code{"cuda_pca"} with \code{loadings} (\code{p x
#'   ncomp}), \code{scores} (\code{n x ncomp}), \code{values} (eigenvalues
#'   \code{||X v||^2}), \code{iters}, and \code{backend} (\code{"cuda"} or
#'   \code{"cpu-fallback"}).
#' @export
cuda_pca <- function(x, ncomp = 1L, max_iter = 200L, tol = 1e-10) {
  x <- as.matrix(x)
  if (!is.numeric(x) || anyNA(x)) {
    stop("'x' must be a numeric matrix without NA", call. = FALSE)
  }
  if (storage.mode(x) != "double") storage.mode(x) <- "double"

  ncomp <- as.integer(ncomp)
  max_iter <- as.integer(max_iter)
  if (length(ncomp) != 1L || is.na(ncomp) || ncomp < 1L || ncomp > ncol(x)) {
    stop("'ncomp' must be between 1 and ncol(x)", call. = FALSE)
  }
  if (length(max_iter) != 1L || is.na(max_iter) || max_iter < 1L) {
    stop("'max_iter' must be a positive integer", call. = FALSE)
  }
  if (length(tol) != 1L || is.na(tol) || tol <= 0) {
    stop("'tol' must be positive", call. = FALSE)
  }

  # cuda_available() is imported from rcupp; referencing it guarantees the
  # rcupp namespace (and its DLL, hence R_init_rcupp registration) is loaded
  # before the C++ R_GetCCallable() lookups below.
  backend <- if (cuda_available()) "cuda" else "cpu-fallback"

  res <- cuda_pca_cpp(x, ncomp, max_iter, as.double(tol))
  res$backend <- backend
  class(res) <- "cuda_pca"
  res
}
