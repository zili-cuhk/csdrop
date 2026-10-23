## comparators.R -- penalized comparators that assume independent censoring.
##   CS_LASSO: cause-specific Cox lasso per margin (event1, event2, dropout)
##   FG_LASSO: Fine-Gray lasso per cause via survival::finegray + glmnet
## Both use unpenalized refits on the selected support for coefficients and CIFs.

suppressPackageStartupMessages({
  library(survival)
  library(glmnet)
})
if (packageVersion("glmnet") < "4.1") stop("glmnet >= 4.1 is required (start-stop Cox)")

CS_CRITS <- c("cv_min", "cv_1se", "bic", "ebic0.5")
FG_CRITS <- c("cv_min", "cv_1se")

## Sum of we over the risk set {start < t <= stop}, for each t
risk_sum <- function(t, start, stop, we) {
  tail_sum <- function(s, cs, t) {
    i <- findInterval(t, s, left.open = TRUE) + 1L   # first index with s >= t
    out <- numeric(length(t))
    k <- i <= length(cs)
    out[k] <- cs[i[k]]
    out
  }
  o1 <- order(stop);  c1 <- rev(cumsum(rev(we[o1])))
  o2 <- order(start); c2 <- rev(cumsum(rev(we[o2])))
  tail_sum(stop[o1], c1, t) - tail_sum(start[o2], c2, t)
}

## Weighted Breslow baseline-hazard jumps and partial log-likelihood
breslow <- function(start, stop, event, w, eta) {
  ev <- event == 1
  if (!any(ev)) return(list(time = numeric(0), dL0 = numeric(0), loglik = 0))
  et  <- sort(unique(stop[ev]))
  den <- risk_sum(et, start, stop, w * exp(eta))
  num <- as.vector(rowsum(w[ev], stop[ev]))         # ordered by sort(unique(.))
  list(time = et, dL0 = num / den,
       loglik = sum(w[ev] * eta[ev]) - sum(num * log(den)))
}

## Unpenalized Cox refit on support S (S may be empty)
cox_refit <- function(X, start, stop, event, w, S) {
  beta <- numeric(ncol(X))
  if (length(S) > 0) {
    XS <- X[, S, drop = FALSE]
    fit <- tryCatch(
      suppressWarnings(coxph(Surv(start, stop, event) ~ XS, weights = w,
                             ties = "breslow",
                             control = coxph.control(iter.max = 50))),
      error = function(e) NULL)
    if (is.null(fit) || any(!is.finite(coef(fit))) || any(abs(coef(fit)) > 20))
      return(NULL)
    beta[S] <- unname(coef(fit))
  }
  bh <- breslow(start, stop, event, w, drop(X %*% beta))
  list(beta = beta, bh = bh, loglik = bh$loglik)
}

## Lasso path + criteria for one margin. Returns a named list over criteria,
## each list(support, beta, bh) or NULL on failure.
GLMNET_TIES <- if (utils::packageVersion("glmnet") >= "5.1") list(cox.ties = "breslow") else list()
fit_margin <- function(X, start, stop, event, w, foldid, n_ic, crits) {
  y <- if (all(start == 0)) Surv(stop, event) else Surv(start, stop, event)
  cv <- tryCatch(do.call(cv.glmnet, c(list(X, y, family = "cox", weights = w, foldid = foldid), GLMNET_TIES)),
                 error = function(e) NULL)
  out <- setNames(vector("list", length(crits)), crits)
  if (is.null(cv)) return(out)
  p <- ncol(X)
  nz <- function(s) which(as.vector(coef(cv, s = s)) != 0)
  for (cr in intersect(crits, c("cv_min", "cv_1se"))) {
    S <- nz(if (cr == "cv_min") "lambda.min" else "lambda.1se")
    rf <- cox_refit(X, start, stop, event, w, S)
    if (!is.null(rf)) out[[cr]] <- c(list(support = S), rf)
  }
  ic <- intersect(crits, c("bic", "ebic0.5"))
  if (length(ic) > 0) {
    Bp <- as.matrix(cv$glmnet.fit$beta)
    max_size <- min(p, max(1, floor(sum(w[event == 1]) / 5)))
    sup <- lapply(seq_len(ncol(Bp)), function(l) which(Bp[, l] != 0))
    sup <- c(list(integer(0)), sup)
    sup <- sup[lengths(sup) <= max_size]
    sup <- sup[!duplicated(vapply(sup, paste, "", collapse = ","))]
    refits <- lapply(sup, function(S) cox_refit(X, start, stop, event, w, S))
    keep <- !vapply(refits, is.null, TRUE)
    sup <- sup[keep]; refits <- refits[keep]
    if (length(sup) > 0) {
      k  <- lengths(sup)
      ll <- vapply(refits, `[[`, 0, "loglik")
      for (cr in ic) {
        g <- if (cr == "bic") 0 else 0.5
        crit <- -2 * ll + k * log(n_ic) + 2 * g * lchoose(p, k)
        j <- order(round(crit, 10), k)[1]           # ties -> smaller support
        out[[cr]] <- c(list(support = sup[[j]]), refits[[j]])
      }
    }
  }
  out
}

## Step-function lookup of cumulative quantities on a grid
on_grid <- function(time, values, grid) {
  idx <- findInterval(grid, time)                    # number of jump times <= grid
  cbind(0, values)[, idx + 1, drop = FALSE]
}

row_cumsum <- function(M) {
  if (ncol(M) > 1) for (j in 2:ncol(M)) M[, j] <- M[, j - 1] + M[, j]
  M
}

## Plug-in CIFs from two cause-specific Cox fits (exp form of S_T)
cs_cif <- function(r1, r2, Z, grid = GRID) {
  tt <- sort(unique(c(r1$bh$time, r2$bh$time)))
  if (length(tt) == 0) {
    zero <- matrix(0, nrow(Z), length(grid)); return(list(zero, zero))
  }
  d1 <- numeric(length(tt)); d1[match(r1$bh$time, tt)] <- r1$bh$dL0
  d2 <- numeric(length(tt)); d2[match(r2$bh$time, tt)] <- r2$bh$dL0
  e1 <- exp(drop(Z %*% r1$beta)); e2 <- exp(drop(Z %*% r2$beta))
  Ltot <- outer(e1, cumsum(d1)) + outer(e2, cumsum(d2))
  Sminus <- exp(-cbind(0, Ltot)[, seq_along(tt), drop = FALSE])   # S_T(t_j-)
  F1 <- row_cumsum(Sminus * outer(e1, d1))
  F2 <- row_cumsum(Sminus * outer(e2, d2))
  list(on_grid(tt, F1, grid), on_grid(tt, F2, grid))
}

## Fine-Gray CIF: 1 - exp(-exp(z'b) Lambda_sub(t))
fg_cif <- function(r, Z, grid = GRID) {
  if (length(r$bh$time) == 0) return(matrix(0, nrow(Z), length(grid)))
  L <- on_grid(r$bh$time, matrix(cumsum(r$bh$dL0), 1), grid)       # 1 x G
  1 - exp(-outer(exp(drop(Z %*% r$beta)), drop(L)))
}

fmt_support <- function(S) paste(S, collapse = ";")
fmt_coef <- function(b) paste(sprintf("%.10g", b), collapse = ";")

## ---- dataset-level runners: return list(sel = data.frame, cif = data.frame)

run_cs_lasso <- function(dat, B, foldid, grid = GRID) {
  X <- dat$Z; n <- nrow(X); p <- ncol(X)
  zero <- rep(0, n); one <- rep(1, n)
  margins <- c(event1 = 1L, event2 = 2L, dropout = 3L)
  fits <- lapply(margins, function(k)
    fit_margin(X, zero, dat$Y, as.integer(dat$status == k), one, foldid, n, CS_CRITS))
  Ft <- true_cif(X, B, grid)
  sel <- cif <- list()
  for (cr in CS_CRITS) {
    for (m in names(margins)) {
      f <- fits[[m]][[cr]]
      sel[[length(sel) + 1]] <- data.frame(
        criterion = cr, margin = m,
        support = if (is.null(f)) "" else fmt_support(f$support),
        coef = if (is.null(f)) "" else fmt_coef(f$beta),
        status = if (is.null(f)) "fit_fail" else "ok")
    }
    f1 <- fits$event1[[cr]]; f2 <- fits$event2[[cr]]
    if (!is.null(f1) && !is.null(f2)) {
      Fh <- cs_cif(f1, f2, X, grid)
      cif[[length(cif) + 1]] <- data.frame(
        criterion = cr, cause = 1:2,
        ise = c(cif_ise(Fh[[1]], Ft[[1]], grid), cif_ise(Fh[[2]], Ft[[2]], grid)))
    }
  }
  list(sel = do.call(rbind, sel), cif = do.call(rbind, cif))
}

run_fg_lasso <- function(dat, B, foldid, grid = GRID) {
  X <- dat$Z; n <- nrow(X); p <- ncol(X)
  zn <- paste0("Z", seq_len(p))
  df <- data.frame(id = seq_len(n), Y = dat$Y,
                   ev = factor(ifelse(dat$status %in% c(0L, 3L), "cens",
                                      paste0("e", dat$status)),
                               levels = c("cens", "e1", "e2")))
  df[zn] <- as.data.frame(X)
  fml <- as.formula(paste("Surv(Y, ev) ~", paste(c("id", zn), collapse = " + ")))
  Ft <- true_cif(X, B, grid)
  sel <- cif <- list()
  for (k in 1:2) {
    fg <- finegray(fml, data = df, etype = paste0("e", k))
    stopifnot(all(c("id", zn, "fgstart", "fgstop", "fgstatus", "fgwt") %in% names(fg)))
    Xf <- as.matrix(fg[, zn])
    fits <- fit_margin(Xf, fg$fgstart, fg$fgstop, fg$fgstatus, fg$fgwt,
                       foldid[fg$id], n, FG_CRITS)
    for (cr in FG_CRITS) {
      f <- fits[[cr]]
      sel[[length(sel) + 1]] <- data.frame(
        criterion = cr, margin = paste0("event", k),
        support = if (is.null(f)) "" else fmt_support(f$support),
        coef = if (is.null(f)) "" else fmt_coef(f$beta),
        status = if (is.null(f)) "fit_fail" else "ok")
      if (!is.null(f)) {
        cif[[length(cif) + 1]] <- data.frame(
          criterion = cr, cause = k,
          ise = cif_ise(fg_cif(f, X, grid), Ft[[k]], grid))
      }
    }
  }
  list(sel = do.call(rbind, sel), cif = do.call(rbind, cif))
}
