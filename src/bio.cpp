#include <Rcpp.h>
#include <rcupp.h>
#include <algorithm>
#include <cmath>
#include <vector>

/*
 * cudaBio: a DOWNSTREAM consumer of the rcupp CUDA-for-Rcpp framework.
 *
 * This file has NO CUDA code and the package has NO configure / Makevars /
 * CUDA detection / CPU-fallback logic. It reaches rcupp's fixed operators
 * purely through <rcupp.h> (supplied by LinkingTo: rcupp -> -I.../include)
 * and R_GetCCallable(). It compiles and produces identical results whether
 * rcupp was built with a CUDA backend or its CPU fallback.
 *
 * Workload: leading eigenvectors of X^T X by power iteration with deflation
 * -- the kernel behind PCA / GWAS on p>>n genomics matrices. Each rcupp_gemv
 * / rcupp_gemv_t call copies host data to the device and back (F0 one-shot
 * ABI); persistent device residency is a later rcupp session API (P1).
 */

namespace {

void init_unit(std::vector<double>& v) {
  const int p = static_cast<int>(v.size());
  for (int j = 0; j < p; ++j) v[j] = std::sin(static_cast<double>(j + 1));
  const double nv = rcupp_nrm2(p, v.data());
  if (nv > 0.0) rcupp_scal(p, 1.0 / nv, v.data());
}

}  // namespace

/*
 * Internal LinkingTo smoke test: exercise every public rcupp operator once so
 * the package R CMD check can compare each result against a pure-R reference.
 * Not exported; called as cudaBio:::rcupp_ops_check_cpp().
 */
// [[Rcpp::export]]
Rcpp::List rcupp_ops_check_cpp(Rcpp::NumericMatrix X, Rcpp::NumericVector v,
                               double t) {
  const int n = X.nrow();
  const int p = X.ncol();
  std::vector<double> g(n), gt(p);
  std::vector<double> a(v.begin(), v.end());
  std::vector<double> s(v.begin(), v.end());
  std::vector<double> st(p);

  rcupp_gemv(X.begin(), n, p, v.begin(), g.data());          /* g = X v        */
  rcupp_gemv_t(X.begin(), n, p, g.data(), gt.data());        /* gt = X^T X v   */
  rcupp_axpy(p, 0.5, v.begin(), a.data());                   /* a = 1.5 v      */
  rcupp_scal(p, 2.0, s.data());                              /* s = 2 v        */
  const double d = rcupp_dot(p, v.begin(), v.begin());
  const double nrm = rcupp_nrm2(p, v.begin());
  rcupp_soft_threshold(v.begin(), t, st.data(), p);

  return Rcpp::List::create(
      Rcpp::Named("gemv")           = g,
      Rcpp::Named("gemv_t")         = gt,
      Rcpp::Named("axpy")           = a,
      Rcpp::Named("scal")           = s,
      Rcpp::Named("dot")            = d,
      Rcpp::Named("nrm2")           = nrm,
      Rcpp::Named("soft_threshold") = st,
      Rcpp::Named("cuda")           = static_cast<bool>(rcupp_cuda_available()));
}

/*
 * ncomp leading eigenvectors of X^T X. X is n x p (column-major); returns
 * loadings (p x ncomp), scores (n x ncomp) and the eigenvalues ||X v||^2.
 */
// [[Rcpp::export]]
Rcpp::List cuda_pca_cpp(Rcpp::NumericMatrix X, int ncomp,
                        int max_iter, double tol) {
  const int n = X.nrow();
  const int p = X.ncol();

  std::vector<double> A(X.begin(), X.end());   /* mutable column-major copy  */
  std::vector<double> loadings(static_cast<std::size_t>(ncomp) * p, 0.0);
  std::vector<double> scores(static_cast<std::size_t>(ncomp) * n, 0.0);
  std::vector<double> values(ncomp, 0.0);
  std::vector<double> v(p), u(n), w(p), vnew(p), s(n);
  int total_iters = 0;

  for (int c = 0; c < ncomp; ++c) {
    init_unit(v);
    for (int it = 0; it < max_iter; ++it) {
      rcupp_gemv(A.data(), n, p, v.data(), u.data());        /* u = A v       */
      rcupp_gemv_t(A.data(), n, p, u.data(), w.data());      /* w = A^T A v   */
      const double nw = rcupp_nrm2(p, w.data());
      if (nw <= 0.0) break;
      for (int j = 0; j < p; ++j) vnew[j] = w[j] / nw;
      double diff = 0.0;
      for (int j = 0; j < p; ++j) {
        const double dd = vnew[j] - v[j];
        diff += dd * dd;
      }
      v.swap(vnew);
      ++total_iters;
      if (std::sqrt(diff) < tol) break;
    }

    /* scores s = A v; eigenvalue = ||A v||^2 for a unit v. */
    rcupp_gemv(A.data(), n, p, v.data(), s.data());
    values[c] = rcupp_dot(n, s.data(), s.data());
    for (int i = 0; i < n; ++i)
      scores[static_cast<std::size_t>(c) * n + i] = s[i];
    for (int j = 0; j < p; ++j)
      loadings[static_cast<std::size_t>(c) * p + j] = v[j];

    /* Deflate: A <- A - s v^T, so the next component is orthogonal. */
    for (int j = 0; j < p; ++j)
      for (int i = 0; i < n; ++i)
        A[i + static_cast<std::size_t>(j) * n] -= s[i] * v[j];
  }

  Rcpp::NumericMatrix L(p, ncomp);
  std::copy(loadings.begin(), loadings.end(), L.begin());
  Rcpp::NumericMatrix S(n, ncomp);
  std::copy(scores.begin(), scores.end(), S.begin());

  return Rcpp::List::create(
      Rcpp::Named("loadings") = L,
      Rcpp::Named("scores")   = S,
      Rcpp::Named("values")   = values,
      Rcpp::Named("iters")    = total_iters);
}
