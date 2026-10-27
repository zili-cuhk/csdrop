## prep_hrs.R -- analysis cohort for the Health and Retirement Study (HRS) application.
##
## Usage (from the repository root):
##   Rscript application/hrs/prep_hrs.R <RAND HRS .dta> <Langa-Weir .dta> [outdir]
## e.g.
##   Rscript application/hrs/prep_hrs.R application/hrs/data/randhrs1992_2022v1.dta \
##           application/hrs/data/cogfinalimp_9520wide.dta application/hrs/data
##
## Inputs (free registration at https://hrsdata.isr.umich.edu; not redistributed):
##   RAND HRS Longitudinal File (Stata), for interview status and dates, death
##     dates and baseline covariates
##   Langa-Weir Classification of Cognitive Function, 1995-2020 (Stata,
##     cogfinalimp_9520wide.dta), for the dementia classification by wave
##
## Design
##   Baseline: the 2000 wave (RAND wave 5). Self-respondents aged 65 or older,
##     classified as not demented in 2000 (cogfunction2000 in {1, 2}), with
##     complete baseline covariates.
##   Time: years since the 2000 interview.
##   Event 1: dementia, the first wave from 2002 to 2020 with cogfunction = 3,
##     at that wave's interview date.
##   Event 2: death without an earlier dementia classification, at the death date.
##   Dropout (status 3): permanently dropped from the sample (RAND RwIWSTAT = 7)
##     or not interviewed in any later wave up to 2022 while not known to have
##     died, censored at the last interview.
##   Administrative censoring (status 0): event-free and interviewed in 2020,
##     the last wave with a dementia classification, censored at that interview.
##   Respondents who missed the 2020 interview but were interviewed again in
##     2022 are administratively censored at their last interview before 2020
##     in the main cohort, and treated as dropout in the sensitivity cohort.
##
## Outputs (in outdir)
##   hrs_main.csv, hrs_sens.csv   columns Y, status, covariates (status 1 dementia,
##                                2 death without dementia, 3 dropout, 0 administrative)
##   hrs_cuts.csv                 internal cutpoints of the baseline partition
##   hrs_summary.txt              cohort flow, event counts, dropout reasons, and
##                                mortality after dropout (a descriptive check of
##                                dependent dropout that uses deaths observed after
##                                the participant left the study)

suppressMessages({
  if (!requireNamespace("haven", quietly = TRUE))
    stop("package haven is needed: install.packages('haven')")
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript prep_hrs.R <RAND .dta> <Langa-Weir .dta> [outdir]")
f_rand <- args[1]; f_lw <- args[2]
outdir <- if (length(args) >= 3) args[3] else dirname(f_rand)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
log <- file(file.path(outdir, "hrs_summary.txt"), "w")
say <- function(...) { s <- sprintf(...); cat(s, "\n"); cat(s, "\n", file = log) }

## ---------------------------------------------------------------- read
W <- 5:16                                    # RAND waves 2000 .. 2022
YEAR <- c(`5` = 2000, `6` = 2002, `7` = 2004, `8` = 2006, `9` = 2008, `10` = 2010,
          `11` = 2012, `12` = 2014, `13` = 2016, `14` = 2018, `15` = 2020, `16` = 2022)
base_vars <- c("hhidpn", "hhid", "pn", "raddate", "ragender", "raracem", "rahispan",
               "raedyrs", "r5agey_b", "r5proxy", "r5mstat", "r5bmi", "r5smoken",
               "r5smokev", "r5drink", "r5hibpe", "r5diabe", "r5cancre", "r5lunge",
               "r5hearte", "r5stroke", "r5psyche", "r5arthre", "r5shlt", "r5adla",
               "r5iadla", "r5cesd", "h5itot", "h5atotb", "r5vigact", "r5hosp")
wave_vars <- c(paste0("r", W, "iwstat"), paste0("r", W, "iwend"))
nm <- names(haven::read_dta(f_rand, n_max = 0))
want <- c(base_vars, wave_vars)
hit <- nm[tolower(nm) %in% want]
miss <- setdiff(want, tolower(hit))
if (length(miss)) say("RAND variables not found (dropped if covariates): %s", paste(miss, collapse = " "))
need <- c("hhid", "pn", "raddate", "r5agey_b", "r5proxy", wave_vars)
if (any(need %in% miss)) stop("required RAND variables missing: ", paste(intersect(need, miss), collapse = " "))
R <- as.data.frame(haven::read_dta(f_rand, col_select = tidyselect::all_of(hit)))
names(R) <- tolower(names(R))
L <- as.data.frame(haven::read_dta(f_lw))
names(L) <- tolower(names(L))
num <- function(x) as.numeric(haven::zap_labels(x))
asdate <- function(x) {
  if (inherits(x, "Date")) return(x)
  as.Date(num(x), origin = "1960-01-01")   # SAS and Stata day counts
}
key <- function(h, p) paste(sprintf("%06d", as.integer(as.character(haven::zap_labels(h)))),
                             sprintf("%03d", as.integer(as.character(haven::zap_labels(p)))))
R$key <- key(R$hhid, R$pn); L$key <- key(L$hhid, L$pn)
D <- merge(R, L[, c("key", grep("^cogfunction|^cogtot27_imp2000$", names(L), value = TRUE))],
           by = "key", all.x = TRUE)
say("RAND records %d, Langa-Weir records %d, merged %d", nrow(R), nrow(L), nrow(D))

## ---------------------------------------------------------------- cohort
n0 <- nrow(D)
D <- D[!is.na(D$r5iwstat) & num(D$r5iwstat) == 1, ];               n1 <- nrow(D)
D <- D[!is.na(D$r5proxy) & num(D$r5proxy) == 0, ];                 n2 <- nrow(D)
D <- D[!is.na(D$r5agey_b) & num(D$r5agey_b) >= 65, ];              n3 <- nrow(D)
D <- D[!is.na(D$cogfunction2000) & num(D$cogfunction2000) %in% c(1, 2), ]; n4 <- nrow(D)
say("cohort flow: all %d -> interviewed 2000 %d -> self-respondent %d -> age >= 65 %d -> not demented %d",
    n0, n1, n2, n3, n4)

## ---------------------------------------------------------------- outcomes
t0 <- asdate(D$r5iwend)
dd <- asdate(D$raddate)
st <- sapply(W, function(w) num(D[[paste0("r", w, "iwstat")]]))
iw <- sapply(W, function(w) as.numeric(asdate(D[[paste0("r", w, "iwend")]])))
cf <- sapply(W, function(w) { v <- paste0("cogfunction", YEAR[as.character(w)])
                              if (v %in% names(D)) num(D[[v]]) else rep(NA_real_, nrow(D)) })
colnames(st) <- colnames(iw) <- colnames(cf) <- W
if (all(is.na(cf[, "7"]))) say("note: no 2004 dementia classification; dementia first seen in 2006 is dated 2006")

outcome <- function(i, temp_nr_as_dropout) {
  last <- as.numeric(t0[i]); reason <- NA_character_
  for (w in as.character(6:15)) {
    s <- st[i, w]
    if (is.na(s)) next
    if (s == 1) {                                      # interviewed, alive
      if (!is.na(cf[i, w]) && cf[i, w] == 3) return(list(t = iw[i, w], s = 1, r = NA))
      if (!is.na(iw[i, w])) last <- iw[i, w]
    } else if (s %in% c(5, 6)) {                       # died this or a previous wave
      t <- as.numeric(dd[i])
      if (is.na(t) || t <= last) t <- last + 0.5 * (if (is.na(iw[i, w])) 730 else iw[i, w] - last)
      return(list(t = t, s = 2, r = NA))
    } else if (s == 7) {                               # dropped from the sample
      return(list(t = last, s = 3, r = "dropped from sample"))
    }                                                  # 4 or 9: missed this wave, may return
  }
  if (!is.na(st[i, "15"]) && st[i, "15"] == 1) return(list(t = last, s = 0, r = NA))
  back2022 <- !is.na(st[i, "16"]) && st[i, "16"] == 1
  if (back2022 && !temp_nr_as_dropout) return(list(t = last, s = 0, r = "missed 2020, back in 2022"))
  list(t = last, s = 3, r = if (back2022) "missed 2020, back in 2022" else "no further interview")
}
build <- function(temp) {
  o <- lapply(seq_len(nrow(D)), outcome, temp_nr_as_dropout = temp)
  data.frame(Y = (sapply(o, `[[`, "t") - as.numeric(t0)) / 365.25,
             status = sapply(o, `[[`, "s"), reason = sapply(o, `[[`, "r"))
}

## ---------------------------------------------------------------- covariates
v <- function(n) if (n %in% names(D)) num(D[[n]]) else rep(NA_real_, nrow(D))
Z <- data.frame(
  age10 = v("r5agey_b") / 10, female = as.numeric(v("ragender") == 2),
  black = as.numeric(v("raracem") == 2), otherrace = as.numeric(v("raracem") == 3),
  hispanic = v("rahispan"), educ = v("raedyrs"),
  married = as.numeric(v("r5mstat") %in% c(1, 2, 3)), bmi = v("r5bmi"),
  smoke_now = v("r5smoken"), smoke_ever = v("r5smokev"), drink = v("r5drink"),
  hibp = v("r5hibpe"), diab = v("r5diabe"), cancer = v("r5cancre"), lung = v("r5lunge"),
  heart = v("r5hearte"), stroke = v("r5stroke"), psych = v("r5psyche"),
  arthr = v("r5arthre"), srh = v("r5shlt"), adl = v("r5adla"), iadl = v("r5iadla"),
  cesd = v("r5cesd"), loginc = log1p(pmax(v("h5itot"), 0)),
  wealth = asinh(v("h5atotb") / 1000), vigact = v("r5vigact"), hosp = v("r5hosp"),
  cog27 = v("cogtot27_imp2000"), cind = as.numeric(v("cogfunction2000") == 2))
Z <- Z[, colSums(!is.na(Z)) > 0, drop = FALSE]          # drop covariates absent from the file
cc <- complete.cases(Z)
say("complete baseline covariates: %d of %d (%d covariates: %s)", sum(cc), nrow(Z), ncol(Z),
    paste(names(Z), collapse = " "))

## ---------------------------------------------------------------- write
lab <- c(`0` = "administrative", `1` = "dementia", `2` = "death without dementia", `3` = "dropout")
for (k in c("main", "sens")) {
  O <- build(k == "sens")[cc, ]
  X <- cbind(O[, c("Y", "status")], Z[cc, ])
  bad <- !is.finite(X$Y) | X$Y <= 0
  if (any(bad)) say("%s: %d records with nonpositive or missing time removed", k, sum(bad))
  X <- X[!bad, ]; O <- O[!bad, ]
  write.csv(X, file.path(outdir, sprintf("hrs_%s.csv", k)), row.names = FALSE)
  tab <- table(factor(X$status, levels = 0:3, labels = lab))
  say("%s cohort: n = %d; %s", k, nrow(X), paste(names(tab), tab, sep = " ", collapse = ", "))
  say("  dropout reasons: %s", paste(names(table(O$reason[X$status == 3])),
                                     table(O$reason[X$status == 3]), collapse = ", "))
  say("  follow-up (years): median %.1f, max %.1f", median(X$Y), max(X$Y))
  if (k == "main") {
    q <- quantile(X$Y[X$status > 0], c(0.2, 0.4, 0.6, 0.8))
    write.table(t(round(q, 4)), file.path(outdir, "hrs_cuts.csv"), sep = ",",
                row.names = FALSE, col.names = FALSE)   # one line, no header
    say("  baseline cutpoints (quintiles of occurrence times): %s", paste(round(q, 2), collapse = " "))
    ## Mortality after censoring, per 100 person-years up to the latest death
    ## date in the file. HRS learns of some deaths after a respondent leaves
    ## (exit interviews, tracker updates), so the dropout rate is a lower bound.
    ## A clearly higher rate after dropout than after administrative censoring
    ## is descriptive evidence that dropout is related to health.
    tdeath <- (as.numeric(dd[cc][!bad]) - as.numeric(t0[cc][!bad])) / 365.25
    tend <- (max(as.numeric(dd), na.rm = TRUE) - as.numeric(t0[cc][!bad])) / 365.25
    for (s in c(3, 0)) {
      g <- X$status == s
      died <- !is.na(tdeath[g]) & tdeath[g] > X$Y[g]
      py <- pmax(ifelse(died, tdeath[g], tend[g]) - X$Y[g], 0)
      say("  %s (n = %d): %d deaths after censoring, %.2f per 100 person-years",
          lab[as.character(s)], sum(g), sum(died), 100 * sum(died) / sum(py))
    }
  }
}
close(log)
cat("written to", outdir, "\n")
