## generate.R -- write simulated datasets to data/<sid>/rep_XXXX.csv
## Usage:
##   Rscript R/generate.R --task K            (LSF array index, chunks of 50)
##   Rscript R/generate.R --task K --exp P    (task index within the scenarios of one experiment)
##   Rscript R/generate.R --sid A01 --from 1 --to 500
##   Rscript R/generate.R --task K --exp P --digits 8
##       (covariates rounded to 8 significant digits, which halves the disk use of
##        the larger-dimension cells; Y is always written with 15 digits)
## Columns: Y, status (1 event1, 2 event2, 3 dropout, 0 admin), Z1..Zp

source("R/dgp.R")
source("R/scenarios.R")
S <- read_scenarios()
args <- parse_args()
if (!is.null(args$exp)) S <- S[S$exp %in% strsplit(args$exp, ",")[[1]], ]

if (!is.null(args$task)) {
  tm <- task_map(as.integer(args$task), S)
  s <- tm$s; from <- tm$from; to <- tm$to
} else {
  s <- match(args$sid, S$sid)
  if (is.na(s)) stop("unknown sid")
  from <- as.integer(args$from); to <- as.integer(args$to)
}
digits <- if (is.null(args$digits)) NA else as.integer(args$digits)
sc <- S[s, ]
dir <- file.path("data", sc$sid)
dir.create(dir, recursive = TRUE, showWarnings = FALSE)

if (from == 1) {
  B <- coefs(sc$signal, sc$p)
  write.csv(data.frame(margin = rownames(B), B), file.path(dir, "truth.csv"),
            row.names = FALSE)
}

man <- vector("list", to - from + 1)
for (r in from:to) {
  seed <- sc$base_seed + r
  set.seed(seed)
  d <- gen_data(sc$n, sc$p, sc$signal, sc$family, sc$tau, sc$mC, sc$a, sc$delta,
                f2 = sc$f2, baseline = sc$baseline, admin = sc$admin)
  Z <- if (is.na(digits)) d$Z else signif(d$Z, digits)
  out <- data.frame(Y = d$Y, status = d$status, Z)
  names(out) <- c("Y", "status", paste0("Z", seq_len(sc$p)))
  f <- file.path(dir, sprintf("rep_%04d.csv", r))
  write.csv(out, f, row.names = FALSE)              # 15 significant digits
  man[[r - from + 1]] <- data.frame(
    sid = sc$sid, rep = r, seed = seed,
    n_event1 = sum(d$status == 1), n_event2 = sum(d$status == 2),
    n_dropout = sum(d$status == 3), n_admin = sum(d$status == 0))
}
write.csv(do.call(rbind, man),
          file.path(dir, sprintf("manifest_%04d_%04d.csv", from, to)),
          row.names = FALSE)
cat(sprintf("%s: wrote reps %d-%d\n", sc$sid, from, to))
