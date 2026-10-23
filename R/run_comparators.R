## run_comparators.R -- fit CS_LASSO and FG_LASSO on saved datasets.
## Usage:
##   Rscript R/run_comparators.R --task K
##   Rscript R/run_comparators.R --sid A01 --from 1 --to 50 [--methods CS_LASSO,FG_LASSO]
## Output: results/comp/{sel,cif}_<fit_id>_<sid>_<from>.csv

source("R/dgp.R")
source("R/scenarios.R")
source("R/comparators.R")
S <- read_scenarios()
args <- parse_args()

if (!is.null(args$task)) {
  tm <- task_map(as.integer(args$task), S)
  s <- tm$s; from <- tm$from; to <- tm$to
} else {
  s <- match(args$sid, S$sid)
  if (is.na(s)) stop("unknown sid")
  from <- as.integer(args$from); to <- as.integer(args$to)
}
methods <- if (is.null(args$methods)) c("CS_LASSO", "FG_LASSO") else
  strsplit(args$methods, ",")[[1]]
sc <- S[s, ]
B <- coefs(sc$signal, sc$p)
options(csdrop.f2 = sc$f2, csdrop.baseline = sc$baseline)   # read by true_cif()
outdir <- file.path("results", "comp")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

runners <- list(CS_LASSO = run_cs_lasso, FG_LASSO = run_fg_lasso)
acc <- setNames(lapply(methods, function(m) list(sel = list(), cif = list())), methods)

for (r in from:to) {
  d <- read.csv(file.path("data", sc$sid, sprintf("rep_%04d.csv", r)))
  dat <- list(Y = d$Y, status = as.integer(d$status), Z = as.matrix(d[, -(1:2)]))
  set.seed(sc$base_seed + r + 50000000)               # fold seed, separate from data seed
  foldid <- sample(rep(1:10, length.out = nrow(d)))
  for (m in methods) {
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(runners[[m]](dat, B, foldid), error = function(e) e)
    secs <- proc.time()[["elapsed"]] - t0
    if (inherits(res, "error")) {
      acc[[m]]$sel[[length(acc[[m]]$sel) + 1]] <- data.frame(
        sid = sc$sid, rep = r, fit_id = m, criterion = "ALL", margin = "ALL",
        support = "", coef = "", status = paste("error:", conditionMessage(res)),
        seconds = secs)
      next
    }
    acc[[m]]$sel[[length(acc[[m]]$sel) + 1]] <-
      cbind(sid = sc$sid, rep = r, fit_id = m, res$sel, seconds = secs)
    if (!is.null(res$cif))
      acc[[m]]$cif[[length(acc[[m]]$cif) + 1]] <-
        cbind(sid = sc$sid, rep = r, fit_id = m, res$cif)
  }
}

for (m in methods) {
  tag <- sprintf("%s_%s_%04d.csv", m, sc$sid, from)
  write.csv(do.call(rbind, acc[[m]]$sel), file.path(outdir, paste0("sel_", tag)),
            row.names = FALSE)
  if (length(acc[[m]]$cif) > 0)
    write.csv(do.call(rbind, acc[[m]]$cif), file.path(outdir, paste0("cif_", tag)),
              row.names = FALSE)
}
cat(sprintf("%s reps %d-%d done (%s)\n", sc$sid, from, to, paste(methods, collapse = ",")))
