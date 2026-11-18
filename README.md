# cudaBio

<img src="man/figures/logo.png" align="right" height="160" alt="cudaBio hexsticker" />

[![Check](https://github.com/shajoezhu/cudaBio/actions/workflows/check.yaml/badge.svg)](https://github.com/shajoezhu/cudaBio/actions/workflows/check.yaml)

A minimal **downstream example of the [rcupp](https://github.com/shajoezhu/rcupp)
CUDA-for-Rcpp framework**, for bioinformatics-style workloads. It computes
principal components by **power iteration** on `p >> n` matrices (PCA / GWAS
style), calling only `rcupp`'s fixed GPU operators from its own C++ through
`LinkingTo: rcupp` and `R_GetCCallable()`.

The point of this package is the build story: it contains **no `.cu` files, no
`configure`, no `src/Makevars`, and no CUDA detection / CPU-fallback logic** of
its own. It compiles and produces identical results whether `rcupp` was built
with a real CUDA backend or its CPU fallback. Use it as a template for writing
your own GPU-accelerated R package without taking on any CUDA build complexity.

## What it provides

- `cuda_pca(x, ncomp = 1, max_iter = 200, tol = 1e-10)` — leading eigenvectors
  of `crossprod(x)` by power iteration (with deflation for `ncomp > 1`).
  Returns a `cuda_pca` object with `loadings` (`p x ncomp`), `scores`
  (`n x ncomp`), `values` (eigenvalues `||X v||^2`), `iters`, and `backend`
  (`"cuda"` or `"cpu-fallback"`).

Under the hood the C++ calls `rcupp_gemv` / `rcupp_gemv_t` (the `X^T X v`
product), `rcupp_dot` / `rcupp_nrm2` / `rcupp_scal` for normalization and
convergence, and `rcupp_soft_threshold` is covered by the operator self-check.
All of them are host-pointer, one-shot operators resolved at runtime.

## Installation

`rcupp` must be installed first (any backend):

```r
# install.packages("remotes")
remotes::install_github("shajoezhu/rcupp")
remotes::install_github("shajoezhu/cudaBio")
```

No CUDA Toolkit is needed to build `cudaBio` itself — acceleration comes from
whatever `rcupp` was compiled against.

## Quick start

```r
library(cudaBio)
set.seed(1)
X <- matrix(rnorm(50 * 200), 50, 200)   # p >> n, e.g. genotypes

pc <- cuda_pca(X, ncomp = 2)
dim(pc$loadings)   # 200 x 2
dim(pc$scores)     #  50 x 2
pc$values          # leading eigenvalues
pc$backend         # "cuda" or "cpu-fallback"
```

## How it links to rcupp (template for your own package)

```
# DESCRIPTION
Imports: rcupp, Rcpp
LinkingTo: rcupp, Rcpp
NeedsCompilation: yes
```

```
# NAMESPACE
useDynLib(myPkg, .registration = TRUE)
importFrom(rcupp, cuda_available)
importFrom(Rcpp, evalCpp)
```

```cpp
#include <Rcpp.h>
#include <rcupp.h>

// inside your .Call-able function:
rcupp_gemv(X.begin(), n, p, v.begin(), out.data());   // out = X %*% v
```

`LinkingTo: rcupp` adds `-I<rcupp>/include` automatically (no `src/Makevars`).
The `static inline` wrappers in `<rcupp.h>` call `R_GetCCallable("rcupp", ...)`,
so there is no link-time dependency on `rcupp.so` — just make sure to reference
something from `rcupp` (e.g. `cuda_available()`) in R before the C++ runs, so
its namespace and DLL registrations are loaded. Regenerate exports with
`Rcpp::compileAttributes("myPkg")` after editing — `R CMD build/check` does not
run it for you.

See the `getting-started` vignette and
[rcupp's public API](https://github.com/shajoezhu/rcupp) for the full operator
and session C ABI.

## Part of the cudaGlmnet family

| Package | Role | Link |
|---|---|---|
| rcupp | low-level Rcpp/CUDA bridge + reusable operator/session framework | [github](https://github.com/shajoezhu/rcupp) |
| cudaGlmnet | glmnet-style GPU lasso front end | [github](https://github.com/shajoezhu/cudaGlmnet) |
| **cudaBio** | PCA / GWAS power iteration (`p >> n`) downstream example | this repo |
| cudaFin | rolling-window lasso forecasting downstream example | [github](https://github.com/shajoezhu/cudaFin) |

## License

Apache License 2.0.
