function res = csm_select(Y, status, Z, cuts, K, family, tau, opt)
%CSM_SELECT Algorithm 1: GSELO continuation paths, thresholding, support-
%restricted refits and EBIC selection for a fixed copula (family, tau).
%
%   For each gamma_P, three candidates are computed for every shape theta:
%     inc  increasing continuation path over V, started from zero coefficients
%          with occurrence/exposure baseline heights;
%     dec  decreasing continuation path, same start;
%     mle  fixed start at the full-model unpenalized joint estimate.
%   Along a path the next shape is warm-started from the previous fit, also
%   after a failed fit. Every penalized fit uses opt.n_starts starts (the
%   given start and a jittered copy). For each (gamma_P, theta), the admissible
%   candidate with an admissible refit and the smallest objective is retained.
%
%   res = csm_select(Y, status, Z, cuts, K, family, tau)
%   res = csm_select(..., opt)              % opt from csm_options()
%
%   status : 1..K event type, K+1 dropout, 0 administrative censoring
%   Z      : n x p raw covariates (standardized internally)
%
%   res.status  'ok' | 'nonestimable' | 'no_admissible_refit'
%   res.P       prepared data (standardization constants, cuts, ...)
%   res.sel     1 x numel(opt.gE) struct array, one per EBIC coefficient:
%                 gE, gP, theta, S ((K+1) x p logical), qR, ell, ebic,
%                 X, xi (standardized-scale refit), B, a (raw-scale refit), src
%   res.cand    table of retained (gP, theta) candidates with refit criteria
%   res.mle_status  status of the full-model unpenalized fit
%   res.n_failed    number of penalized candidates that were not admissible
if nargin < 8 || isempty(opt), opt = csm_options(); end
P = csm_prep(Y, status, Z, cuts, K);
res = struct('status', '', 'P', P, 'sel', [], 'cand', [], 'mle_status', '', 'n_failed', 0);
if any(P.n0 < opt.min_events) || any(P.Dm(:) < 1)
    res.status = 'nonestimable'; return
end
cop = struct('family', family, 'theta', csm_tau2theta(family, tau));
D1 = K + 1; p = P.p; M = P.M;
nopen = struct('kind', 'none');
X0 = zeros(D1, p);

% ---- full-model unpenalized estimate (fixed start of the mle candidates)
mle = csm_fit(P, cop, X0, csm_breslow_init(P, X0), nopen, [], opt, true);
res.mle_status = mle.status;
useMle = all(isfinite(mle.X(:))) && isfinite(mle.Q);
xiCrude = crude_init(P);

% ---- penalized candidates and retained solution per (gamma_P, theta)
nG = numel(opt.gP_grid); nV = numel(opt.V);
rows = struct('gP', {}, 'theta', {}, 'S', {}, 'qR', {}, 'ell', {}, 'X', {}, 'xi', {}, ...
    'src', {}, 'Qpen', {});
cacheKeys = {}; cacheFits = {};
nFail = 0;
for g = 1:nG
    rho = log(max(P.n0, 2)) + 2 * opt.gP_grid(g) * log(D1 * p);   % eq. (rho)
    cands = repmat({{}}, nV, 1);
    cid = 0;
    for dirn = 1:2
        if dirn == 1, order = 1:nV; src = 'inc'; else, order = nV:-1:1; src = 'dec'; end
        Xc = X0; xic = xiCrude;
        for v = order
            cid = cid + 1;
            pen = struct('kind', 'gselo', 'rho', rho, 'theta', opt.V(v));
            f = fit_multistart(P, cop, Xc, xic, pen, opt, opt.jitter_seed + 1000 * (g - 1) + 10 * cid);
            f.src = src;
            cands{v}{end + 1} = f;
            Xc = f.X; xic = f.xi;          % warm start, also after a failed fit
        end
    end
    for v = 1:nV
        cid = cid + 1;
        if ~useMle, continue, end
        pen = struct('kind', 'gselo', 'rho', rho, 'theta', opt.V(v));
        f = fit_multistart(P, cop, mle.X, mle.xi, pen, opt, opt.jitter_seed + 1000 * (g - 1) + 10 * cid);
        f.src = 'mle';
        cands{v}{end + 1} = f;
    end
    for v = 1:nV
        bestQ = Inf; best = [];
        for c = 1:numel(cands{v})
            f = cands{v}{c};
            if ~strcmp(f.status, 'ok'), nFail = nFail + 1; continue, end
            S = abs(f.X) > opt.eta;
            key = char('0' + S(:)');
            j = find(strcmp(cacheKeys, key), 1);
            if isempty(j)
                rf = csm_fit(P, cop, f.X, f.xi, nopen, S, opt, true);
                cacheKeys{end + 1} = key; cacheFits{end + 1} = rf; %#ok<AGROW>
            else
                rf = cacheFits{j};
            end
            if ~strcmp(rf.status, 'ok'), continue, end
            if f.Q < bestQ
                bestQ = f.Q;
                best = struct('gP', opt.gP_grid(g), 'theta', opt.V(v), 'S', S, 'qR', nnz(S), ...
                    'ell', rf.ell, 'X', rf.X, 'xi', rf.xi, 'src', f.src, 'Qpen', f.Q);
            end
        end
        if ~isempty(best), rows(end + 1) = best; end %#ok<AGROW>
    end
end
res.n_failed = nFail;
if isempty(rows)
    res.status = 'no_admissible_refit'; return
end

% ---- EBIC with the prespecified tie rule
Pdim = D1 * p; n = P.n;
qR = [rows.qR]; ellv = [rows.ell];
lchoose = gammaln(Pdim + 1) - gammaln(qR + 1) - gammaln(Pdim - qR + 1);
sel = struct([]);
crit = zeros(numel(opt.gE), numel(rows));
for e = 1:numel(opt.gE)
    c = -2 * ellv + (qR + D1 * M) * log(n) + 2 * opt.gE(e) * lchoose;
    crit(e, :) = c;
    bi = 1;
    for j = 2:numel(rows)
        tol = opt.tie_rel * abs(c(bi));
        if c(j) < c(bi) - tol
            bi = j;
        elseif abs(c(j) - c(bi)) <= tol
            kj = [rows(j).qR, rows(j).gP, rows(j).theta, c(j)];
            kb = [rows(bi).qR, rows(bi).gP, rows(bi).theta, c(bi)];
            if lexless(kj, kb), bi = j; end
        end
    end
    r = rows(bi);
    [B, a] = csm_to_raw(P, r.X, r.xi);
    s = struct('gE', opt.gE(e), 'gP', r.gP, 'theta', r.theta, 'S', r.S, 'qR', r.qR, ...
        'ell', r.ell, 'ebic', c(bi), 'X', r.X, 'xi', r.xi, 'B', B, 'a', a, 'src', r.src);
    if isempty(sel), sel = s; else, sel(e) = s; end %#ok<AGROW>
end
res.sel = sel;
res.cand = table([rows.gP]', [rows.theta]', {rows.src}', [rows.Qpen]', qR', ellv', crit', ...
    'VariableNames', {'gP', 'theta', 'src', 'Qpen', 'qR', 'ell', 'ebic'});
res.status = 'ok';
end

function f = fit_multistart(P, cop, X0, xi0, pen, opt, seed)
% Penalized fit from X0 and from jittered copies; returns the admissible fit
% with the smallest objective (or the smallest finite one if none is admissible).
fits = cell(opt.n_starts, 1);
fits{1} = csm_fit(P, cop, X0, xi0, pen, [], opt, false);
for s = 2:opt.n_starts
    Xj = min(max(X0 + opt.jitter_sd * jitter(size(X0), seed + s - 2), -opt.coef_bound), opt.coef_bound);
    fits{s} = csm_fit(P, cop, Xj, xi0, pen, [], opt, false);
end
Q = cellfun(@(x) x.Q, fits);
okv = cellfun(@(x) strcmp(x.status, 'ok'), fits);
if any(okv)
    Q(~okv) = Inf;
else
    Q(~isfinite(Q)) = Inf;
end
[~, k] = min(Q);
f = fits{k};
end

function z = jitter(sz, seed)
% Standard normal draws shared with the Python reference implementation.
st = rng;
rng(seed, 'twister');
u = rand(sz);
rng(st);
z = sqrt(2) * erfinv(2 * u - 1);
end

function xi = crude_init(P)
% Occurrence/exposure baseline heights, floored at 1e-3.
expo = sum(P.E, 1);
xi = log(max(P.Dm ./ max(expo, 1e-12), 1e-3));
end

function tf = lexless(a, b)
% lexicographic a < b
k = find(a ~= b, 1);
tf = ~isempty(k) && a(k) < b(k);
end
