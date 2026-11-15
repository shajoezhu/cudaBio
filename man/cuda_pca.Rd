\name{cuda_pca}
\alias{cuda_pca}
\title{Principal components by GPU power iteration (rcupp framework example)}
\usage{
cuda_pca(x, ncomp = 1L, max_iter = 200L, tol = 1e-10)
}
\arguments{
  \item{x}{Numeric matrix \code{n x p}.}
  \item{ncomp}{Number of leading components (between 1 and \code{ncol(x)}).}
  \item{max_iter}{Maximum power-iteration steps per component.}
  \item{tol}{Convergence tolerance on the change of the unit eigenvector.}
}
\value{
A list of class \code{"cuda_pca"} with \code{loadings} (\code{p x ncomp}),
\code{scores} (\code{n x ncomp}), \code{values} (eigenvalues), \code{iters},
and \code{backend}.
}
\description{
Computes the leading eigenvectors of \code{crossprod(x)} by power iteration,
calling the fixed GPU operators exposed by \pkg{rcupp} through
\code{LinkingTo: rcupp} and \code{R_GetCCallable()}. This package ships no CUDA
code and no CUDA build logic of its own; it works identically whether
\pkg{rcupp} was built with a CUDA backend or its CPU fallback.
}
