function res = csm_thr_mle(Y, status, Z, cuts, K, family, tau, opt)
%CSM_THR_MLE No-GSELO comparator: admissible full-model unpenalized joint MLE
%(lowest objective over the zero / marginal / weak-ridge seeds), thresholded at
%opt.eta on the standardized scale, then refitted on the retained support.
%   Output fields match one element of csm_select's res.sel (gE, gP, theta = NaN).
if nargin < 8 || isempty(opt), opt = csm_options(); end
P = csm_prep(Y, status, Z, cuts, K);
res = struct('status', '', 'P', P, 'sel', []);
if any(P.n0 < opt.min_events) || any(P.Dm(:) < 1)
    res.status = 'nonestimable'; return
end
cop = struct('family', family, 'theta', csm_tau2theta(family, tau));
nopen = struct('kind', 'none');
indep = struct('family', 'indep', 'theta', 0);
X0 = zeros(K + 1, P.p);
fm = csm_fit(P, indep, X0, csm_breslow_init(P, X0), nopen, [], opt, true);
fr = csm_fit(P, indep, X0, csm_breslow_init(P, X0), ...
    struct('kind', 'ridge', 'lam', opt.ridge_lam), [], opt, true);
seeds = {X0, fm.X, fr.X};
bestf = [];
for s = 1:3
    f = csm_fit(P, cop, seeds{s}, csm_breslow_init(P, seeds{s}), nopen, [], opt, true);
    if strcmp(f.status, 'ok') && (isempty(bestf) || f.Q < bestf.Q)
        bestf = f;
    end
end
if isempty(bestf)
    res.status = 'no_admissible_branch'; return
end
S = abs(bestf.X) > opt.eta;
rf = csm_fit(P, cop, bestf.X, bestf.xi, nopen, S, opt, true);
if ~strcmp(rf.status, 'ok')
    res.status = 'no_admissible_refit'; return
end
[B, a] = csm_to_raw(P, rf.X, rf.xi);
res.sel = struct('gE', NaN, 'gP', NaN, 'theta', NaN, 'S', S, 'qR', nnz(S), ...
    'ell', rf.ell, 'ebic', NaN, 'X', rf.X, 'xi', rf.xi, 'B', B, 'a', a);
res.status = 'ok';
end
