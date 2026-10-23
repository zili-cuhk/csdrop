## dgp.R -- data-generating mechanism of Supplement S4 (manuscript v13).
## Mirrors a Python port that reproduced the paper's reported event, dropout and
## administrative-censoring rates with the frozen calibration constants.
##
## Status codes: 1 = event 1, 2 = event 2, 3 = dropout, 0 = administrative.

CUT <- c(0, 1, 2, Inf)
A1  <- c(0.120, 0.090, 0.070)            # event-1 baseline heights
A2  <- c(0.090, 0.075, 0.060)            # event-2 baseline heights
AC0 <- c(0.055, 0.075, 0.100)            # dropout heights before the multiplier mC
GRID <- seq(0, 6, by = 0.1)              # CIF evaluation grid used in the paper
## Weibull event baselines of the Study 2 baseline-misspecification cell:
## Lambda_k(t) = W_k t^KAP, matching the piecewise cumulative hazards at t = 2.
KAP <- 1.5
W1  <- 0.21 / 2^1.5
W2  <- 0.165 / 2^1.5

## Frozen calibration constants for the Frank cells (Supplement S4)
FROZEN <- data.frame(
  signal = rep(c("moderate", "strong"), each = 3),
  tau    = rep(c(0, 0.2, 0.5), 2),
  mC     = c(0.550843413027010, 0.679796401819543, 0.919451899640470,
             0.521141378895718, 0.612753275222009, 0.770079873784685),
  a      = c(8.76565595085491, 9.39301248954755, 10.10206933249560,
             7.36516463793110, 7.84234197037593, 8.39305286737204),
  stringsAsFactors = FALSE
)

## True regression coefficients: rows event1, event2, dropout
coefs <- function(signal, p) {
  if (p < 7) stop("p must be at least 7")
  B <- matrix(0, 3, p, dimnames = list(c("event1", "event2", "dropout"), NULL))
  if (signal == "moderate") {
    B[1, c(1, 2, 5)] <- c(0.50, -0.40, 0.30)
    B[2, c(1, 3, 6)] <- c(-0.35, 0.45, -0.30)
    B[3, c(2, 4, 7)] <- c(0.40, -0.35, 0.30)
  } else if (signal == "strong") {
    B[1, c(1, 2, 5)] <- c(0.70, -0.70, 0.60)
    B[2, c(1, 3, 6)] <- c(-0.70, 0.70, -0.60)
    B[3, c(2, 4, 7)] <- c(0.70, -0.70, 0.60)
  } else {
    stop("unknown signal: ", signal)
  }
  B
}

ar1_sigma <- function(p, rho = 0.3) rho^abs(outer(seq_len(p), seq_len(p), "-"))

## Kendall's tau -> copula parameter
tau2theta <- function(family, tau) {
  if (family == "frank") {
    if (tau == 0) return(0)
    debye1 <- function(th) {
      f <- function(t) ifelse(t == 0, 1, t / expm1(t))
      integrate(f, 0, th, rel.tol = 1e-12)$value / th
    }
    g <- function(th) 1 - 4 / th + 4 * debye1(th) / th - tau
    rng <- if (tau > 0) c(1e-6, 200) else c(-200, -1e-6)
    return(uniroot(g, rng, tol = 1e-13)$root)
  }
  if (family == "clayton") return(2 * tau / (1 - tau))
  if (family == "gumbel")  return(1 / (1 - tau))
  stop("unknown family: ", family)
}

## Draw (U, V) from the copula. U plays the role of S_T(T) and V of S_C(C),
## so drawing from C gives the survival copula used in the likelihood.
rcop <- function(family, theta, n) {
  u <- runif(n)
  w <- runif(n)
  if (family == "frank") {
    if (theta == 0) return(cbind(u, w))
    v <- -log1p(w * expm1(-theta) / (w + (1 - w) * exp(-theta * u))) / theta
    return(cbind(u, v))
  }
  if (family == "clayton") {
    if (theta == 0) return(cbind(u, w))
    v <- ((w^(-theta / (1 + theta)) - 1) * u^(-theta) + 1)^(-1 / theta)
    return(cbind(u, v))
  }
  if (family == "gumbel") {
    al <- 1 / theta
    if (al == 1) return(cbind(u, w))
    ## Marshall-Olkin: positive stable frailty (Chambers-Mallows-Stuck)
    W <- runif(n) * pi
    E <- rexp(n)
    S <- sin(al * W) / sin(W)^(1 / al) * (sin((1 - al) * W) / E)^((1 - al) / al)
    E1 <- rexp(n)
    E2 <- rexp(n)
    return(cbind(exp(-(E1 / S)^al), exp(-(E2 / S)^al)))
  }
  stop("unknown family: ", family)
}

## Invert a piecewise-constant cumulative hazard on cutpoints (0, 1, 2, Inf).
## h: target cumulative hazard (length n); R: n x 3 matrix of interval rates.
inv_cumhaz <- function(h, R) {
  c1 <- R[, 1]
  c2 <- c1 + R[, 2]
  ifelse(h <= c1, h / R[, 1],
         ifelse(h <= c2, 1 + (h - c1) / R[, 2], 2 + (h - c2) / R[, 3]))
}

## Latent quantities that do not depend on (mC, a). Used by both the generator
## and the calibration (common random numbers).
gen_latent <- function(n, p, signal, family, tau, delta = 0, Z = NULL,
                       f2 = 1, baseline = "pc", admin = "fixed") {
  ## f2 multiplies the event-2 baseline (rare competing event), baseline is
  ## "pc" (piecewise constant, the primary design) or "weibull", and admin is
  ## "fixed" (common administrative time) or "stagger" (uniform entry).
  ## With the defaults the random-number stream is that of the primary design.
  if (is.null(Z)) Z <- matrix(rnorm(n * p), n, p) %*% chol(ar1_sigma(p))
  B  <- coefs(signal, p)
  e1 <- exp(drop(Z %*% B[1, ]))
  e2 <- exp(drop(Z %*% B[2, ]))
  eC <- exp(drop(Z %*% B[3, ]))
  UV <- rcop(family, tau2theta(family, tau), n)
  if (baseline == "pc") {
    T  <- inv_cumhaz(-log(UV[, 1]), outer(e1, A1) + outer(e2, f2 * A2))
    m  <- findInterval(T, CUT)                     # interval index 1..3
    p1 <- e1 * A1[m] / (e1 * A1[m] + f2 * e2 * A2[m])
  } else if (baseline == "weibull") {
    cc <- W1 * e1 + f2 * W2 * e2
    T  <- (-log(UV[, 1]) / cc)^(1 / KAP)
    p1 <- W1 * e1 / cc
  } else stop("unknown baseline: ", baseline)
  J  <- ifelse(runif(n) < p1, 1L, 2L)
  ## delta != 0 makes dropout depend on the event type (violates J indep C | T, Z)
  RC <- outer(eC * exp(delta * (J == 1L)), AC0)
  ## staggered entry: administrative time a * (1/2 + V/2) with V uniform
  Vs <- if (admin == "stagger") runif(n) else rep(1, n)
  list(Z = Z, T = T, J = J, H = -log(UV[, 2]), RC = RC, B = B, Vs = Vs, admin = admin)
}

## Administrative times for a given scale a
admin_time <- function(L, a) if (L$admin == "stagger") a * (0.5 + 0.5 * L$Vs) else rep(a, length(L$T))

## Observed data for given dropout multiplier mC and administrative time a
gen_data <- function(n, p, signal, family, tau, mC, a, delta = 0,
                     f2 = 1, baseline = "pc", admin = "fixed") {
  L <- gen_latent(n, p, signal, family, tau, delta, f2 = f2, baseline = baseline, admin = admin)
  C <- inv_cumhaz(L$H / mC, L$RC)                  # rates mC * AC0 * exp(...)
  A <- admin_time(L, a)
  Y <- pmin(L$T, C, A)
  status <- ifelse(L$T <= C & L$T <= A, L$J, ifelse(C <= A, 3L, 0L))
  list(Z = L$Z, Y = Y, status = as.integer(status), B = L$B)
}

## Exact true CIFs on a grid for every row of Z. Returns list(F1, F2), n x G.
true_cif <- function(Z, B, grid = GRID, f2 = getOption("csdrop.f2", 1),
                     baseline = getOption("csdrop.baseline", "pc")) {
  e1 <- exp(drop(Z %*% B[1, ]))
  e2 <- exp(drop(Z %*% B[2, ])) * f2
  n  <- nrow(Z)
  if (baseline == "weibull") {
    ## common shape: F_k(t) = (W_k e_k / r) {1 - exp(-r t^KAP)}, r = W1 e1 + W2 e2
    r <- W1 * e1 + W2 * e2
    inc <- 1 - exp(-outer(r, grid^KAP))
    return(list((W1 * e1 / r) * inc, (W2 * e2 / r) * inc))
  }
  F1 <- F2 <- matrix(0, n, length(grid))
  Sprev <- rep(1, n)                                # S_T at the start of the interval
  for (m in 1:3) {
    lo <- CUT[m]; hi <- CUT[m + 1]
    r1 <- A1[m] * e1; r2 <- A2[m] * e2; r <- r1 + r2
    d   <- pmin(pmax(grid - lo, 0), hi - lo)
    inc <- 1 - exp(-outer(r, d))
    F1 <- F1 + (Sprev * r1 / r) * inc               # vector recycles over rows
    F2 <- F2 + (Sprev * r2 / r) * inc
    if (is.finite(hi)) Sprev <- Sprev * exp(-r * (hi - lo))
  }
  list(F1, F2)
}

## Integrated squared CIF error: trapezoid over the grid, divided by its
## length, averaged over subjects (definition used in the paper).
cif_ise <- function(Fhat, Ftrue, grid = GRID) {
  sq <- (Fhat - Ftrue)^2
  G  <- length(grid)
  per <- ((sq[, -1, drop = FALSE] + sq[, -G, drop = FALSE]) / 2) %*% diff(grid)
  mean(per) / (max(grid) - min(grid))
}

## Map an LSF array index to (scenario row, rep range). Same rule as MATLAB.
task_map <- function(task, S, chunk = 50) {
  nch <- ceiling(S$R / chunk)
  cum <- cumsum(nch)
  s <- which(task <= cum)[1]
  if (is.na(s)) stop("task index ", task, " exceeds ", max(cum))
  c0 <- task - c(0, cum)[s] - 1
  from <- c0 * chunk + 1
  list(s = s, from = from, to = min(S$R[s], from + chunk - 1))
}

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    out[[key]] <- args[i + 1]
    i <- i + 2
  }
  out
}
