function f = csm_fit(P, cop, X0, xi0, pen, mask, opt, unpen)
%CSM_FIT Alternating majorization / backtracking solver of Sections 3.3-3.6.
%   Coefficient block: closed-form update (22) with backtracking on L.
%   Baseline block:    diagonally scaled log-height step (24) with line search.
%   mask  : (K+1) x p logical of free coefficients ([] = all); others fixed at 0.
%   unpen : true -> refit stopping rule (single-iteration gradient tolerance,
%           refit_max_iter); false -> penalized rule (n_consec iterations).
%   f.status is 'ok' or a failure label; only 'ok' fits are admissible.
n = P.n;
if isempty(mask), mask = true(size(X0)); end
X = X0 .* mask;
xi = xi0;
[Q, ell] = objective(P, cop, X, xi, pen);
f = struct('status', 'max_iter', 'X', X, 'xi', xi, 'Q', Q, 'ell', ell, 'iters', 0);
if ~isfinite(Q), f.status = 'nonfinite_start'; return, end
L = 1;
consec = 0;
if unpen, maxIter = opt.refit_max_iter; else, maxIter = opt.max_iter; end
status = 'max_iter';
for it = 1:maxIter
    [ell, gX, ~, okg] = csm_loglik(P, cop, X, xi, true);
    if ~okg, status = 'nonfinite_grad'; break, end
    gg = (-2 / n * gX) .* mask;            % gradient of g = -2 ell / n
    g0 = -2 * ell / n;
    [~, w] = csm_penalty(P, X, pen);
    % ---- coefficient block, eq. (22)
    L = max(L / opt.L_relax, opt.L_floor);
    acc = false;
    for bt = 0:opt.max_bt
        Xn = ((L * X - gg) ./ (L + 2 * w)) .* mask;
        [Qn, elln] = objective(P, cop, Xn, xi, pen);
        dX = Xn - X;
        if isfinite(Qn) && ...
                -2 * elln / n <= g0 + sum(gg .* dX, 'all') + L / 2 * sum(dX .^ 2, 'all') + opt.mono && ...
                Qn <= Q + opt.mono
            acc = true; break
        end
        L = 2 * L;
    end
    if ~acc, status = 'coef_linesearch'; break, end
    dXmax = max(abs(dX), [], 'all');
    X = Xn; Qx = Qn;
    if max(abs(X), [], 'all') > opt.coef_bound, status = 'coef_bound'; break, end
    % ---- baseline block, eq. (24)
    [~, ~, gxi, okg, R] = csm_loglik(P, cop, X, xi, true);
    if ~okg, status = 'nonfinite_grad'; break, end
    a = exp(xi);
    qd = -2 / n * gxi;                                   % = 2(aR - D)/n
    hd = 2 * (P.Dm + abs(a .* R)) / n + opt.eps_num;
    s = 1; acc = false;
    for bt = 0:opt.max_bt
        xin = xi - s * qd ./ hd;
        Qn = objective(P, cop, X, xin, pen);
        if isfinite(Qn) && Qn <= Qx + opt.mono
            acc = true; break
        end
        s = s / 2;
    end
    if ~acc, status = 'base_linesearch'; break, end
    xi = xin;
    a = exp(xi);
    if any(a(:) < opt.a_lo) || any(a(:) > opt.a_hi), status = 'base_bound'; break, end
    Qold = Q; Q = Qn;
    % ---- convergence diagnostics at the new point (Section 3.6)
    [~, gX, gxi, okg] = csm_loglik(P, cop, X, xi, true);
    if ~okg, status = 'nonfinite_grad'; break, end
    [~, w] = csm_penalty(P, X, pen);
    gcoef = max(abs((-2 / n * gX + 2 * w .* X) .* mask), [], 'all');
    gbase = max(abs(-2 / n * gxi), [], 'all');
    if unpen
        if gcoef <= opt.refit_tol && gbase <= opt.refit_tol
            status = 'ok'; break
        end
    else
        cOK = abs(Q - Qold) / max(1, abs(Qold)) <= opt.tol_obj && ...
            dXmax / max(1, max(abs(X), [], 'all')) <= opt.tol_coef && ...
            gcoef <= opt.tol_grad && gbase <= opt.tol_grad;
        if cOK, consec = consec + 1; else, consec = 0; end
        if consec >= opt.n_consec
            status = 'ok'; break
        end
    end
end
ell = csm_loglik(P, cop, X, xi, false);
f = struct('status', status, 'X', X, 'xi', xi, 'Q', Q, 'ell', ell, 'iters', it);
end

function [Q, ell] = objective(P, cop, X, xi, pen)
ell = csm_loglik(P, cop, X, xi, false);
Q = -2 * ell / P.n + csm_penalty(P, X, pen);
end
