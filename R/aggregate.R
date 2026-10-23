## aggregate.R -- combine MATLAB and R results into summary tables.
## Usage: Rscript R/aggregate.R
## Writes results/summary/{selection_by_cell, cif_by_cell, expA_tau_grid,
##                         method_comparison}.csv

source("R/dgp.R")
source("R/scenarios.R")
S <- read_scenarios()
plan <- read.csv("fit_plan.csv", stringsAsFactors = FALSE)
outdir <- file.path("results", "summary")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

read_all <- function(prefix) {
  ## only the current fit folders; backups or summaries elsewhere under results/ are ignored
  f <- list.files(file.path("results", c("prop", "comp")),
                  pattern = paste0("^", prefix, "_.*\\.csv$"), full.names = TRUE)
  if (length(f) == 0) stop("no ", prefix, " files under results/")
  cc <- if (prefix == "sel") c(support = "character", coef = "character") else NA
  out <- do.call(rbind, lapply(f, function(x) {
    d <- read.csv(x, stringsAsFactors = FALSE, colClasses = cc)
    d[, intersect(names(d), c("sid", "rep", "fit_id", "criterion", "margin",
                              "support", "coef", "status", "cause", "ise"))]
  }))
  ## guard against the same fit being written twice by overlapping reruns
  k <- intersect(c("sid", "rep", "fit_id", "criterion", "margin", "cause"), names(out))
  out[!duplicated(out[k]), ]
}

## tau_fit actually used by each fit in each scenario ("true" resolved)
fit_tau <- function(fit_id, sid) {
  sc <- S[match(sid, S$sid), ]
  pl <- plan[match(paste(sc$exp, fit_id), paste(plan$exp, plan$fit_id)), ]
  ifelse(is.na(pl$tau_fit) | pl$tau_fit == "NA", NA,
         ifelse(pl$tau_fit == "true", sc$tau, suppressWarnings(as.numeric(pl$tau_fit))))
}

## ---------------- selection metrics ----------------
sel <- read_all("sel")
sel <- sel[sel$fit_id != "FG_LASSO", ]   # subdistribution coefficients: not comparable
sel$support[is.na(sel$support)] <- ""
sel$ok <- sel$status == "ok"

met <- matrix(NA_real_, nrow(sel), 5, dimnames = list(NULL, c("pcorr", "fp", "fn", "size", "mse")))
for (sid in unique(sel$sid)) {
  sc <- S[S$sid == sid, ]
  B <- coefs(sc$signal, sc$p)
  Sig <- ar1_sigma(sc$p)
  rows <- which(sel$sid == sid & sel$ok)
  for (i in rows) {
    b0 <- B[sel$margin[i], ]
    s  <- if (sel$support[i] == "") integer(0) else as.integer(strsplit(sel$support[i], ";")[[1]])
    t0 <- which(b0 != 0)
    bh <- as.numeric(strsplit(sel$coef[i], ";")[[1]])
    d  <- bh - b0
    met[i, ] <- c(setequal(s, t0), length(setdiff(s, t0)), length(setdiff(t0, s)),
                  length(s), drop(crossprod(d, Sig %*% d)))
  }
}
sel <- cbind(sel, met)
## per-replicate metrics, used by R/make_tables.R for paired Monte Carlo SEs
write.csv(sel[sel$criterion != "ALL",
              c("sid", "rep", "fit_id", "criterion", "margin", "status",
                "pcorr", "fp", "fn", "size", "mse")],
          file.path(outdir, "sel_rep.csv"), row.names = FALSE)

key <- c("sid", "fit_id", "criterion", "margin")
agg_sel <- aggregate(cbind(pcorr, fp, fn, size, mse) ~ sid + fit_id + criterion + margin,
                     data = sel[sel$ok, ], FUN = mean)
cnt <- aggregate(data.frame(n_rows = rep(1, nrow(sel)), n_ok = as.numeric(sel$ok)),
                 sel[key], FUN = sum)
agg_sel <- merge(agg_sel, cnt, by = key, all = TRUE)
agg_sel <- agg_sel[agg_sel$criterion != "ALL", ]

## Datasets with any failure, per scenario and fit (whole-dataset errors are
## written with criterion = "ALL", so they are counted here, not in n_rows).
fails <- sel[!sel$ok, c("sid", "rep", "fit_id", "status")]
fails <- fails[!duplicated(fails[c("sid", "rep", "fit_id")]), ]
write.csv(fails, file.path(outdir, "failures.csv"), row.names = FALSE)
if (nrow(fails) > 0) {
  cat("datasets with a failure, by fit:\n")
  print(table(fails$fit_id))
}
agg_sel <- merge(S[, c("sid", "exp", "n", "p", "signal", "family", "tau", "delta")],
                 agg_sel, by = "sid")
agg_sel$tau_fit <- fit_tau(agg_sel$fit_id, agg_sel$sid)
write.csv(agg_sel, file.path(outdir, "selection_by_cell.csv"), row.names = FALSE)

## ---------------- CIF error ----------------
cif <- read_all("cif")
write.csv(cif, file.path(outdir, "cif_rep.csv"), row.names = FALSE)
agg_cif <- aggregate(ise ~ sid + fit_id + criterion + cause, data = cif, FUN = mean)
cif_key <- c("sid", "fit_id", "criterion", "cause")
agg_cif <- merge(agg_cif,
                 aggregate(data.frame(n = rep(1, nrow(cif))), cif[cif_key], FUN = sum),
                 by = cif_key)
agg_cif <- merge(S[, c("sid", "exp", "n", "p", "signal", "family", "tau", "delta")],
                 agg_cif, by = "sid")
agg_cif$tau_fit <- fit_tau(agg_cif$fit_id, agg_cif$sid)
write.csv(agg_cif, file.path(outdir, "cif_by_cell.csv"), row.names = FALSE)

## ---------------- Exp A: truth tau x fitted tau (proposed method) ----------------
## Unweighted average of the cell means over n, p and signal, as in the paper.
PROP_CRIT <- "EBIC0.25"
a <- agg_cif[agg_cif$exp == "A" & grepl("^PROP_T", agg_cif$fit_id) &
               agg_cif$criterion == PROP_CRIT, ]
if (nrow(a) > 0) {
  grid_tab <- aggregate(ise ~ cause + tau + tau_fit, data = a, FUN = mean)
  grid_tab <- reshape(grid_tab, idvar = c("cause", "tau"), timevar = "tau_fit",
                      direction = "wide")
  names(grid_tab) <- sub("^ise\\.", "tau_fit=", names(grid_tab))
  write.csv(grid_tab[order(grid_tab$cause, grid_tab$tau), ],
            file.path(outdir, "expA_tau_grid.csv"), row.names = FALSE)
} else {
  cat("no proposed-method CIF results yet; skipping expA_tau_grid.csv\n")
}

## ---------------- method comparison by truth ----------------
## "PROP_CORRECT" = the PROP_T fit whose tau_fit equals the true tau (Exp A).
label <- function(d) {
  lab <- paste(d$fit_id, d$criterion, sep = ":")
  corr <- d$exp == "A" & grepl("^PROP_T", d$fit_id) & !is.na(d$tau_fit) &
    abs(d$tau_fit - d$tau) < 1e-12
  lab[corr] <- paste0("PROP_CORRECT:", d$criterion[corr])
  lab
}
truth_lab <- function(d) sprintf("%s tau=%.1f delta=%.1f", d$family, d$tau, d$delta)

agg_cif$method <- label(agg_cif); agg_cif$truth <- truth_lab(agg_cif)
agg_sel$method <- label(agg_sel); agg_sel$truth <- truth_lab(agg_sel)
mc_cif <- aggregate(ise ~ exp + truth + method + cause, data = agg_cif, FUN = mean)
mc_cif <- reshape(mc_cif, idvar = c("exp", "truth", "method"), timevar = "cause",
                  direction = "wide")
mc_sel <- aggregate(cbind(pcorr, fp, fn, mse) ~ exp + truth + method + margin,
                    data = agg_sel, FUN = mean)
mc_sel <- reshape(mc_sel, idvar = c("exp", "truth", "method"), timevar = "margin",
                  direction = "wide")
mc <- merge(mc_cif, mc_sel, by = c("exp", "truth", "method"), all = TRUE)
write.csv(mc[order(mc$exp, mc$truth, mc$method), ],
          file.path(outdir, "method_comparison.csv"), row.names = FALSE)
cat("summaries written to", outdir, "\n")
