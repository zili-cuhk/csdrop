## calibrate.R -- calibrate (mC, a) for the Exp B and Study 2 cells and write scenarios.csv.
## Targets: target_C dropout and target_A administrative censoring (20% and 15% by default).
## Python-port reference values for Study 2 (expect about +/- 1% in mC):
##   S01 dropout 10%  mC=0.497 a=11.48   S02 dropout 35%  mC=1.798 a=7.75
##   S03 stagger 15%  mC=0.922 a=13.77   S04 stagger 30%  mC=1.217 a=7.52
##   S05 tau=-0.3     mC=0.429 a=7.66    S06 f2=0.2 (rare event 2, ~10%)
##   S07 Weibull      mC=1.386 a=5.18
## Usage: Rscript R/calibrate.R
##
## Reference values from the Python port (different RNG, so expect agreement
## to roughly +/- 1% in mC and +/- 0.1 in a):
##   moderate clayton 0.5          mC=0.8768 a=10.619
##   moderate gumbel  0.5          mC=0.9894 a=9.587
##   moderate frank   0.2 delta=.5 mC=0.5211 a=9.259
##   moderate frank   0.2 delta=1  mC=0.3791 a=9.291
##   strong   clayton 0.5          mC=0.7428 a=8.569
##   strong   gumbel  0.5          mC=0.8265 a=8.045
##   strong   frank   0.2 delta=.5 mC=0.4782 a=7.729
##   strong   frank   0.2 delta=1  mC=0.3537 a=7.754

source("R/dgp.R")
source("R/scenarios.R")

calibrate_cell <- function(signal, family, tau, delta, n = 2e5, seed = 2026,
                           target_C = 0.20, target_A = 0.15,
                           f2 = 1, baseline = "pc", admin = "fixed") {
  set.seed(seed)
  ## p = 10 is enough: the active variables are among the first 7, and the
  ## AR(1) design gives them the same joint law for any p >= 7.
  L <- gen_latent(n, 10, signal, family, tau, delta, f2 = f2, baseline = baseline, admin = admin)
  if (admin == "stagger") {
    ## nested search: for each mC, the scale a that gives target_A, then mC
    ## that gives target_C (both rates are monotone in their own argument)
    a_for <- function(mC) {
      C <- inv_cumhaz(L$H / mC, L$RC)
      M <- pmin(L$T, C)
      uniroot(function(a) mean(admin_time(L, a) < M) - target_A,
              c(1e-3, 1e3), tol = 1e-10)$root
    }
    drop_rate <- function(mC) {
      a <- a_for(mC)
      C <- inv_cumhaz(L$H / mC, L$RC)
      mean(C < L$T & C < admin_time(L, a))
    }
    mC <- uniroot(function(m) drop_rate(m) - target_C, c(0.02, 50), tol = 1e-10)$root
    return(c(mC = mC, a = a_for(mC)))
  }
  rates <- function(mC) {
    C <- inv_cumhaz(L$H / mC, L$RC)
    M <- pmin(L$T, C)
    a <- unname(quantile(M, 1 - target_A))          # P(min(T, C) > a) = target_A
    c(dropout = mean(C < L$T & C < a), a = a)
  }
  mC <- uniroot(function(m) rates(m)[["dropout"]] - target_C,
                c(0.02, 50), tol = 1e-10)$root
  c(mC = mC, a = rates(mC)[["a"]])
}

check_rates <- function(row, n = 1e6, seed = 2027) {
  set.seed(seed)
  d <- gen_data(n, 10, row$signal, row$family, row$tau, row$mC, row$a, row$delta,
                f2 = row$f2, baseline = row$baseline, admin = row$admin)
  sapply(c(event1 = 1, event2 = 2, dropout = 3, admin = 0),
         function(k) mean(d$status == k))
}

S <- build_scenarios()
## Keep the calibration of cells already in scenarios.csv (same sid and design),
## so that rerunning this script after the revision only calibrates new cells.
if (file.exists("scenarios.csv")) {
  old <- read.csv("scenarios.csv", stringsAsFactors = FALSE)
  key <- function(D) paste(D$sid, D$n, D$p, D$signal, D$family, D$tau, D$delta)
  j <- match(key(S), key(old))
  keep <- !is.na(j) & !is.na(old$mC[j])
  S$mC[keep] <- old$mC[j[keep]]
  S$a[keep]  <- old$a[j[keep]]
  cat("kept the calibration of", sum(keep), "existing cells\n")
}
todo <- which(is.na(S$mC))
for (i in todo) {
  cal <- calibrate_cell(S$signal[i], S$family[i], S$tau[i], S$delta[i],
                        target_C = S$target_C[i], target_A = S$target_A[i],
                        f2 = S$f2[i], baseline = S$baseline[i], admin = S$admin[i])
  S$mC[i] <- cal[["mC"]]
  S$a[i]  <- cal[["a"]]
  r <- check_rates(S[i, ])
  cat(sprintf("%s %-8s %-7s tau=%.1f delta=%.1f f2=%.2f %s %s  mC=%.4f a=%.3f | rates e1 e2 C A %s\n",
              S$sid[i], S$signal[i], S$family[i], S$tau[i], S$delta[i],
              S$f2[i], S$baseline[i], S$admin[i],
              S$mC[i], S$a[i], paste(sprintf("%.4f", r), collapse = " ")))
}
write.csv(S, "scenarios.csv", row.names = FALSE)
cat("wrote scenarios.csv with", nrow(S), "scenarios\n")
