## scenarios.R -- scenario table for Experiments A and B.

build_scenarios <- function(R = 500) {
  ## Exp A: the primary grid, ordered as the paper's tables (n, p, signal, tau)
  A <- expand.grid(tau = c(0, 0.2, 0.5), signal = c("moderate", "strong"),
                   p = c(10, 40), n = c(400, 600), stringsAsFactors = FALSE)
  A <- data.frame(exp = "A", n = A$n, p = A$p, signal = A$signal,
                  family = "frank", tau = A$tau, delta = 0,
                  stringsAsFactors = FALSE)
  key <- match(paste(A$signal, A$tau), paste(FROZEN$signal, FROZEN$tau))
  A$mC <- FROZEN$mC[key]
  A$a  <- FROZEN$a[key]

  ## Exp B: misspecification cells, calibrated by calibrate.R
  truth <- data.frame(family = c("clayton", "gumbel", "frank", "frank"),
                      tau    = c(0.5, 0.5, 0.2, 0.2),
                      delta  = c(0, 0, 0.5, 1.0), stringsAsFactors = FALSE)
  Bg <- expand.grid(k = seq_len(nrow(truth)), signal = c("moderate", "strong"),
                    stringsAsFactors = FALSE)
  B <- data.frame(exp = "B", n = 600, p = 10, signal = Bg$signal,
                  family = truth$family[Bg$k], tau = truth$tau[Bg$k],
                  delta = truth$delta[Bg$k], mC = NA_real_, a = NA_real_,
                  stringsAsFactors = FALSE)

  S <- rbind(A, B)
  S$f2 <- 1; S$baseline <- "pc"; S$admin <- "fixed"
  S$target_C <- 0.20; S$target_A <- 0.15

  ## ---- cells added for the revision (appended, so the sids and seeds of
  ## the rows above are unchanged)
  ## Study 1: n = 1000 with the primary grid (sids A25-A36)
  A2 <- expand.grid(tau = c(0, 0.2, 0.5), signal = c("moderate", "strong"),
                    p = c(10, 40), n = 1000, stringsAsFactors = FALSE)
  A2 <- data.frame(exp = "A", n = A2$n, p = A2$p, signal = A2$signal,
                   family = "frank", tau = A2$tau, delta = 0, stringsAsFactors = FALSE)
  key <- match(paste(A2$signal, A2$tau), paste(FROZEN$signal, FROZEN$tau))
  A2$mC <- FROZEN$mC[key]; A2$a <- FROZEN$a[key]
  ## Study 1: larger dimension (sid H01)
  H <- data.frame(exp = "H", n = 1000, p = 100, signal = "moderate", family = "frank",
                  tau = 0.5, delta = 0, stringsAsFactors = FALSE)
  key <- match(paste(H$signal, H$tau), paste(FROZEN$signal, FROZEN$tau))
  H$mC <- FROZEN$mC[key]; H$a <- FROZEN$a[key]
  N <- rbind(A2, H)
  N$f2 <- 1; N$baseline <- "pc"; N$admin <- "fixed"; N$target_C <- 0.20; N$target_A <- 0.15
  ## Study 2: one factor at a time from n = 600, p = 10, moderate, Frank tau = 0.5
  ## (the base case is the existing cell A15); sids S01-S07
  f <- data.frame(
    tau      = c(0.5,  0.5,  0.5,       0.5,       -0.3,  0.5,  0.5),
    f2       = c(1,    1,    1,         1,         1,     0.2,  1),
    baseline = c("pc", "pc", "pc",      "pc",      "pc",  "pc", "weibull"),
    admin    = c("fixed", "fixed", "stagger", "stagger", "fixed", "fixed", "fixed"),
    target_C = c(0.10, 0.35, 0.20,      0.20,      0.20,  0.20, 0.20),
    target_A = c(0.15, 0.15, 0.15,      0.30,      0.15,  0.15, 0.15),
    stringsAsFactors = FALSE)
  S2 <- data.frame(exp = "S", n = 600, p = 10, signal = "moderate", family = "frank",
                   tau = f$tau, delta = 0, mC = NA_real_, a = NA_real_,
                   f2 = f$f2, baseline = f$baseline, admin = f$admin,
                   target_C = f$target_C, target_A = f$target_A, stringsAsFactors = FALSE)
  S <- rbind(S, N, S2)

  S$sid <- sprintf("%s%02d", S$exp, ave(seq_len(nrow(S)), S$exp, FUN = seq_along))
  S$R <- R
  S$base_seed <- 20261010 + 100000 * seq_len(nrow(S))
  S[, c("sid", "exp", "n", "p", "signal", "family", "tau", "delta",
        "mC", "a", "R", "base_seed", "f2", "baseline", "admin", "target_C", "target_A")]
}

read_scenarios <- function(path = "scenarios.csv") {
  if (!file.exists(path)) stop("scenarios.csv not found; run R/calibrate.R first")
  S <- read.csv(path, stringsAsFactors = FALSE)
  ## scenarios.csv written before the revision lacks the Study 2 columns
  if (is.null(S$f2)) S$f2 <- 1
  if (is.null(S$baseline)) S$baseline <- "pc"
  if (is.null(S$admin)) S$admin <- "fixed"
  if (is.null(S$target_C)) S$target_C <- 0.20
  if (is.null(S$target_A)) S$target_A <- 0.15
  if (anyNA(S$mC) || anyNA(S$a)) stop("scenarios.csv has uncalibrated cells")
  S
}
