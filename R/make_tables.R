## make_tables.R -- manuscript tables with paired Monte Carlo standard errors.
## Run after R/aggregate.R (needs results/summary/cif_rep.csv and sel_rep.csv).
## Usage: Rscript R/make_tables.R
## Writes results/summary/tables/:
##   cif_cell.csv, cif_pooled.csv, sel_cell.csv, sel_pooled.csv   (all methods/criteria)
##   tabA_cif.tex     Exp A: CIF error by true tau, with paired difference vs correct fit
##   tabA_taugrid.tex Exp A: true tau x fitted tau (proposed method)
##   tabA_sel.tex     Exp A: correct-selection rates by margin
##   tabB.tex         Exp B: misspecification
## Pooled values are unweighted averages of the cell means over n, p and signal
## (as in the paper); their SE is sqrt(sum of cell SE^2) / (number of cells).

source("R/dgp.R")
source("R/scenarios.R")
S <- read_scenarios()
sdir <- file.path("results", "summary")
odir <- file.path(sdir, "tables")
dir.create(odir, recursive = TRUE, showWarnings = FALSE)
CRIT <- "EBIC0.25"                      # criterion reported for the proposed method
SC <- S[, c("sid", "exp", "n", "p", "signal", "family", "tau", "delta")]

se_fun <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 2) NA_real_ else sd(x) / sqrt(length(x))
}
mean_fun <- function(x) mean(x, na.rm = TRUE)
pool_se <- function(x) sqrt(sum(x^2, na.rm = TRUE)) / sum(!is.na(x))

## the correctly specified proposed fit for each scenario
ref_method <- function(exp, tau) {
  ifelse(exp == "A", sprintf("PROP_T%02d:%s", as.integer(round(tau * 10)), CRIT),
         paste0("PROP_ORACLE:", CRIT))
}

## cell summaries (by sid) and pooled summaries (over n, p, signal)
summarise <- function(d, vars, keys) {
  mf <- aggregate(d[vars], d[c("sid", keys)], FUN = mean_fun)
  sf <- aggregate(d[vars], d[c("sid", keys)], FUN = se_fun)
  nf <- aggregate(data.frame(nrep = rep(1, nrow(d))), d[c("sid", keys)], FUN = sum)
  names(sf)[names(sf) %in% vars] <- paste0(names(sf)[names(sf) %in% vars], "_se")
  cell <- merge(merge(mf, sf, by = c("sid", keys)), nf, by = c("sid", keys))
  cell <- merge(SC, cell, by = "sid")
  gk <- c("exp", "family", "tau", "delta", keys)
  pm <- aggregate(cell[vars], cell[gk], FUN = mean_fun)
  ps <- aggregate(cell[paste0(vars, "_se")], cell[gk], FUN = pool_se)
  pc <- aggregate(data.frame(ncell = rep(1, nrow(cell))), cell[gk], FUN = sum)
  pooled <- merge(merge(pm, ps, by = gk), pc, by = gk)
  list(cell = cell, pooled = pooled)
}

## ---------------- CIF error ----------------
cif <- read.csv(file.path(sdir, "cif_rep.csv"), stringsAsFactors = FALSE)
cif$method <- paste(cif$fit_id, cif$criterion, sep = ":")
cif <- merge(SC[, c("sid", "exp", "tau")], cif, by = "sid")
cif$ise <- 1000 * cif$ise                                   # report x 10^3
refs <- cif[cif$method == ref_method(cif$exp, cif$tau), c("sid", "rep", "cause", "ise")]
names(refs)[4] <- "ise_ref"
cif <- merge(cif, refs, by = c("sid", "rep", "cause"), all.x = TRUE)
cif <- cif[!is.na(cif$ise_ref), ]                  # pair only reps where the reference fit exists
cif$diff <- cif$ise - cif$ise_ref
cif$exp <- NULL; cif$tau <- NULL                            # re-attached by summarise()
CS <- summarise(cif, c("ise", "diff"), c("method", "cause"))
write.csv(CS$cell, file.path(odir, "cif_cell.csv"), row.names = FALSE)
write.csv(CS$pooled, file.path(odir, "cif_pooled.csv"), row.names = FALSE)

## ---------------- selection ----------------
sel <- read.csv(file.path(sdir, "sel_rep.csv"), stringsAsFactors = FALSE)
sel <- sel[sel$status == "ok", ]
sel$method <- paste(sel$fit_id, sel$criterion, sep = ":")
SS <- summarise(sel, c("pcorr", "fp", "fn", "mse"), c("method", "margin"))
write.csv(SS$cell, file.path(odir, "sel_cell.csv"), row.names = FALSE)
write.csv(SS$pooled, file.path(odir, "sel_pooled.csv"), row.names = FALSE)

## ---------------- lookups and formatting ----------------
pkey <- function(exp, family, tau, delta, method, x)
  paste(exp, family, sprintf("%.2f", tau), sprintf("%.2f", delta), method, x, sep = "|")
CP <- CS$pooled
CP$key <- pkey(CP$exp, CP$family, CP$tau, CP$delta, CP$method, CP$cause)
SP <- SS$pooled
SP$key <- pkey(SP$exp, SP$family, SP$tau, SP$delta, SP$method, SP$margin)

get_row <- function(P, exp, family, tau, delta, method, x) {
  i <- match(pkey(exp, family, tau, delta, method, x), P$key)
  if (is.na(i)) NULL else P[i, ]
}
fmt <- function(m, se, digits = 2) {
  if (is.null(m) || is.na(m)) return("--")
  if (is.na(se)) return(formatC(m, format = "f", digits = digits))
  sprintf("%s (%s)", formatC(m, format = "f", digits = digits),
          formatC(se, format = "f", digits = digits))
}
fmt_diff <- function(m, se) {
  if (is.null(m) || is.na(m)) return("--")
  if (abs(m) < 5e-13) return("0")
  if (is.na(se)) return(sprintf("%+.2f", m))
  sprintf("%+.2f (%.2f)", m, se)
}
write_tex <- function(df, file, caption, label, note = NULL) {
  esc_hdr <- names(df)
  body <- apply(as.matrix(df), 1, function(r) paste(r, collapse = " & "))
  lines <- c("\\begin{table}[ht]", "\\centering", "\\small",
             sprintf("\\caption{%s}", caption), sprintf("\\label{%s}", label),
             sprintf("\\begin{tabular}{l%s}", strrep("r", ncol(df) - 1)),
             "\\toprule", paste0(paste(esc_hdr, collapse = " & "), " \\\\"), "\\midrule",
             paste0(body, " \\\\"), "\\bottomrule", "\\end{tabular}",
             if (!is.null(note)) sprintf("\\par\\smallskip{\\footnotesize %s}", note),
             "\\end{table}")
  writeLines(lines, file)
}

TAUS <- c(0, 0.2, 0.5)
CORRECT <- "CORRECT"                    # placeholder resolved per tau
resolve <- function(method, exp, tau) if (method == CORRECT) ref_method(exp, tau) else method

## ---------------- Table A: CIF error ----------------
methA <- c("CORRECT"                    = "Proposed, correct $\\tau$",
           "PROP_T00:EBIC0.25"          = "Proposed, working independence",
           "PROP_LUMP:EBIC0.25"         = "Proposed, administrative lumped with dropout",
           "THR_MLE:THR"                = "Unpenalized MLE, threshold 0.05",
           "THR_MLE_010:THR"            = "Unpenalized MLE, threshold 0.10",
           "THR_MLE_020:THR"            = "Unpenalized MLE, threshold 0.20",
           "THR_MLE_030:THR"            = "Unpenalized MLE, threshold 0.30",
           "CS_LASSO:bic"               = "Cause-specific lasso, BIC refit",
           "CS_LASSO:cv_min"            = "Cause-specific lasso, CV",
           "FG_LASSO:cv_min"            = "Fine--Gray lasso, CV")
rows <- list()
for (m in names(methA)) {
  r <- c(Method = methA[[m]])
  for (tau in TAUS) for (k in 1:2) {
    g <- get_row(CP, "A", "frank", tau, 0, resolve(m, "A", tau), k)
    r[sprintf("$\\tau=%.1f$, cause %d", tau, k)] <-
      if (is.null(g)) "--" else sprintf("%s [%s]", fmt(g$ise, g$ise_se), fmt_diff(g$diff, g$diff_se))
  }
  rows[[length(rows) + 1]] <- r
}
tabA <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
write_tex(tabA, file.path(odir, "tabA_cif.tex"),
          "Integrated squared CIF error ($\\times 10^{3}$), Experiment A.",
          "tab:expA-cif",
          paste("Entries: mean (Monte Carlo SE) [paired difference from the correctly specified",
                "proposed fit (SE)]. Averages of 8 scenario means (n, p, signal); 500 replicates each.",
                paste0("Proposed fits use ", CRIT, ".")))

## ---------------- Table A: tau grid ----------------
rows <- list()
for (tau in TAUS) for (k in 1:2) {
  r <- c("True $\\tau$" = sprintf("%.1f (cause %d)", tau, k))
  for (tf in TAUS) {
    g <- get_row(CP, "A", "frank", tau, 0, sprintf("PROP_T%02d:%s", as.integer(round(tf * 10)), CRIT), k)
    r[sprintf("Fitted $\\tau=%.1f$", tf)] <-
      if (is.null(g)) "--" else sprintf("%s [%s]", fmt(g$ise, g$ise_se), fmt_diff(g$diff, g$diff_se))
  }
  rows[[length(rows) + 1]] <- r
}
tabG <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
write_tex(tabG, file.path(odir, "tabA_taugrid.tex"),
          "Integrated squared CIF error ($\\times 10^{3}$) by true and fitted Kendall's $\\tau$ (Frank copula).",
          "tab:expA-taugrid",
          "Entries: mean (SE) [paired difference from the fit at the true $\\tau$ (SE)].")

## ---------------- Table A: selection ----------------
selA <- methA[names(methA) != "FG_LASSO:cv_min"]
rows <- list()
for (m in names(selA)) {
  r <- c(Method = selA[[m]])
  for (tau in TAUS) for (mg in c("event1", "event2", "dropout")) {
    g <- get_row(SP, "A", "frank", tau, 0, resolve(m, "A", tau), mg)
    lab <- c(event1 = "E1", event2 = "E2", dropout = "D")[[mg]]
    r[sprintf("$\\tau=%.1f$ %s", tau, lab)] <-
      if (is.null(g)) "--" else formatC(g$pcorr, format = "f", digits = 3)
  }
  rows[[length(rows) + 1]] <- r
}
tabS <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
write_tex(tabS, file.path(odir, "tabA_sel.tex"),
          "Proportion of exact support recovery by margin, Experiment A.",
          "tab:expA-sel",
          paste("E1, E2: event margins; D: dropout margin. Largest Monte Carlo SE of a pooled",
                sprintf("proportion: %.3f.", max(SP$pcorr_se[SP$exp == "A"], na.rm = TRUE)),
                "False-positive and false-negative counts are in sel\\_pooled.csv."))

## ---------------- Table B: misspecification ----------------
truthB <- list(list("clayton", 0.5, 0, "Clayton, $\\tau=0.5$"),
               list("gumbel", 0.5, 0, "Gumbel, $\\tau=0.5$"),
               list("frank", 0.2, 0.5, "Frank $\\tau=0.2$, $\\delta=0.5$"),
               list("frank", 0.2, 1.0, "Frank $\\tau=0.2$, $\\delta=1$"))
methB <- c("PROP_ORACLE:EBIC0.25" = "Proposed, true family and $\\tau$",
           "PROP_FRANK:EBIC0.25"  = "Proposed, Frank at true $\\tau$",
           "PROP_T00:EBIC0.25"    = "Proposed, working independence",
           "CS_LASSO:bic"         = "Cause-specific lasso, BIC refit",
           "FG_LASSO:cv_min"      = "Fine--Gray lasso, CV")
rows <- list()
for (tr in truthB) for (m in names(methB)) {
  g1 <- get_row(CP, "B", tr[[1]], tr[[2]], tr[[3]], m, 1)
  if (is.null(g1)) next                     # e.g. PROP_FRANK is not fitted in delta cells
  g2 <- get_row(CP, "B", tr[[1]], tr[[2]], tr[[3]], m, 2)
  s <- lapply(c("event1", "event2", "dropout"),
              function(mg) get_row(SP, "B", tr[[1]], tr[[2]], tr[[3]], m, mg))
  pc <- vapply(s, function(g) if (is.null(g)) "--" else formatC(g$pcorr, format = "f", digits = 3), "")
  rows[[length(rows) + 1]] <- c(Truth = tr[[4]], Method = methB[[m]],
                                "CIF 1" = fmt(g1$ise, g1$ise_se), "CIF 2" = fmt(g2$ise, g2$ise_se),
                                "$\\Delta$ CIF 1" = fmt_diff(g1$diff, g1$diff_se),
                                "$\\Delta$ CIF 2" = fmt_diff(g2$diff, g2$diff_se),
                                E1 = pc[1], E2 = pc[2], D = pc[3])
}
tabB <- as.data.frame(do.call(rbind, rows), stringsAsFactors = FALSE)
write_tex(tabB, file.path(odir, "tabB.tex"),
          "Misspecification experiment (Experiment B; $n=600$, $p=10$).",
          "tab:expB",
          paste("CIF: integrated squared error $\\times 10^{3}$, mean (SE), averaged over the two signal",
                "levels. $\\Delta$: paired difference from the proposed fit with the true family and",
                "$\\tau$. E1, E2, D: proportion of exact support recovery."))

## replication check: each Exp A pooled value should average 8 cells of 500 reps
cat("cells per pooled value (Exp A):", paste(sort(unique(CS$pooled$ncell[CS$pooled$exp == "A"])), collapse = ","),
    "| reps per cell:", paste(range(CS$cell$nrep), collapse = "-"), "\n")
## console summary
options(width = 200)
cat("\n== Experiment A: CIF error x1e3 [paired diff vs correct] ==\n"); print(tabA, row.names = FALSE)
cat("\n== Experiment A: tau grid ==\n"); print(tabG, row.names = FALSE)
cat("\n== Experiment A: exact support recovery ==\n"); print(tabS, row.names = FALSE)
cat("\n== Experiment B ==\n"); print(tabB, row.names = FALSE)
cat("\nwrote tables to", odir, "\n")
