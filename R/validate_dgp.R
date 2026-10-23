## validate_dgp.R -- checks the R generator against the paper before any data
## are produced. Usage: Rscript R/validate_dgp.R   (about 1-2 minutes)

source("R/dgp.R")
ok <- TRUE
report <- function(label, pass, detail = "") {
  cat(sprintf("[%s] %s %s\n", if (pass) "PASS" else "FAIL", label, detail))
  if (!pass) ok <<- FALSE
}

## 1. Frank parameters match the frozen values in Supplement S4
th <- c(tau2theta("frank", 0.2), tau2theta("frank", 0.5))
report("Frank theta", all(abs(th - c(1.86088378085859, 5.73628270701997)) < 1e-8),
       sprintf("(%.11f, %.11f)", th[1], th[2]))

## 2. Copula samplers reproduce Kendall's tau
set.seed(1)
for (cfg in list(c("frank", 0.2), c("frank", 0.5), c("clayton", 0.5), c("gumbel", 0.5))) {
  fam <- cfg[1]; tau <- as.numeric(cfg[2])
  UV <- rcop(fam, tau2theta(fam, tau), 4000)
  kt <- cor(UV[, 1], UV[, 2], method = "kendall")
  report(sprintf("Kendall tau %s %.1f", fam, tau), abs(kt - tau) < 0.03,
         sprintf("(%.3f)", kt))
}

## 3. Inverse cumulative hazard round trip
R3 <- matrix(c(0.2, 0.1, 0.05), 5, 3, byrow = TRUE)
tt <- c(0.3, 1, 1.7, 2.5, 10)
cumh <- function(t, r) r[1] * pmin(t, 1) + r[2] * pmin(pmax(t - 1, 0), 1) + r[3] * pmax(t - 2, 0)
h <- sapply(tt, cumh, r = R3[1, ])
report("inverse cumulative hazard", max(abs(inv_cumhaz(h, R3) - tt)) < 1e-12)

## 4. Outcome rates with the frozen constants (1e6 draws per cell) fall in
##    the ranges reported in Supplement S4, widened by 0.002 for MC error.
paper <- rbind(event1 = c(0.3492, 0.3573), event2 = c(0.2952, 0.3031),
               dropout = c(0.1993, 0.2001), admin = c(0.1478, 0.1482))
rates <- matrix(NA, nrow(FROZEN), 4, dimnames = list(NULL, rownames(paper)))
for (i in seq_len(nrow(FROZEN))) {
  set.seed(100 + i)
  d <- gen_data(1e6, 10, FROZEN$signal[i], "frank", FROZEN$tau[i],
                FROZEN$mC[i], FROZEN$a[i])
  rates[i, ] <- sapply(c(1, 2, 3, 0), function(k) mean(d$status == k))
}
print(cbind(FROZEN[, 1:2], round(rates, 4)))
rng <- apply(rates, 2, range)
for (k in rownames(paper)) {
  report(sprintf("rate range %s", k),
         rng[1, k] >= paper[k, 1] - 0.002 && rng[2, k] <= paper[k, 2] + 0.002,
         sprintf("R: %.4f-%.4f  paper: %.4f-%.4f", rng[1, k], rng[2, k],
                 paper[k, 1], paper[k, 2]))
}

## 5. True CIF formula against Monte Carlo at a fixed covariate vector
set.seed(3)
B <- coefs("moderate", 10)
z <- rnorm(10)
n <- 4e5
L <- gen_latent(n, 10, "moderate", "frank", 0, Z = matrix(z, n, 10, byrow = TRUE))
g <- c(0.5, 1, 1.5, 3, 6)
Ft <- true_cif(matrix(z, 1), B, g)
emp1 <- sapply(g, function(t) mean(L$T <= t & L$J == 1))
emp2 <- sapply(g, function(t) mean(L$T <= t & L$J == 2))
report("true CIF vs Monte Carlo",
       max(abs(c(Ft[[1]] - emp1, Ft[[2]] - emp2))) < 0.004,
       sprintf("(max abs diff %.4f)", max(abs(c(Ft[[1]] - emp1, Ft[[2]] - emp2)))))

cat(if (ok) "\nALL CHECKS PASSED\n" else "\nSOME CHECKS FAILED -- do not generate data\n")
quit(status = if (ok) 0 else 1)
