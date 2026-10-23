function out = fit_prop_adapter(dat, opts)
%FIT_PROP_ADAPTER Bridge between run_prop_fits and the csm_* estimator.
%
% INPUT
%   dat.Y, dat.status, dat.Z   status: 1 event 1, 2 event 2, 3 dropout, 0 admin
%   opts.family      'frank' | 'clayton' | 'gumbel'
%   opts.tau         Kendall's tau to fix (0 = independence)
%   opts.penalty     'gselo'   -> Algorithm 1 (csm_select), criteria BIC/EBIC
%                    'none'    -> unpenalized MLE + threshold + refit (csm_thr_mle)
%                    'liu2026' -> the two-margin model of Liu, Wang, Wang and Song
%                                 (Statistica Sinica 36, 999-1022) fitted separately
%                                 for each cause: the cause is the event, and the
%                                 other cause, dropout and administrative censoring
%                                 together form the dependent censoring time,
%                                 linked by the same copula. Same penalty, path and
%                                 criteria as the proposed method. The event rows
%                                 come from the two fits; the dropout row is the
%                                 censoring margin of the cause-1 fit. The CIFs
%                                 plug the two event hazards into the competing-
%                                 risks formula, as a naive user would.
%   opts.threshold   0.05 (standardized scale)
%   opts.gammaE      [0 0.25 0.5]
%   opts.grid        0:0.1:6
%   opts.cuts        optional; default [0 1 2 Inf] (the simulation partition)
%
% OUTPUT (one entry per criterion): see run_prop_fits.m
K = 2;
if isfield(opts, 'cuts'), cuts = opts.cuts; else, cuts = [0 1 2 Inf]; end
opt = csm_options('eta', opts.threshold, 'gE', opts.gammaE);

switch opts.penalty
    case 'gselo'
        res = csm_select(dat.Y, dat.status, dat.Z, cuts, K, opts.family, opts.tau, opt);
        crits = arrayfun(@crit_name, opts.gammaE, 'UniformOutput', false);
    case 'none'
        res = csm_thr_mle(dat.Y, dat.status, dat.Z, cuts, K, opts.family, opts.tau, opt);
        crits = {'THR'};
    case 'liu2026'
        out = fit_liu2026(dat, opts, cuts, opt);
        return
    otherwise
        error('unknown penalty %s', opts.penalty);
end

C = numel(crits);
out.criteria = crits;
out.status = repmat({res.status}, 1, C);
out.support = cell(1, C);
out.coef = cell(1, C);
out.cif = cell(1, C);
if ~strcmp(res.status, 'ok'), return, end
for c = 1:C
    s = res.sel(c);
    out.support{c} = s.S;
    out.coef{c} = s.B;                                    % raw scale, zeros off support
    out.cif{c} = csm_cif(res.P, s.X, s.xi, dat.Z, opts.grid);
end
end

function s = crit_name(g)
% Labels used in the paper's tables (and by R/aggregate.R).
if g == 0
    s = 'BIC';
else
    s = sprintf('EBIC%.2f', g);       % 0.25 -> EBIC0.25, 0.5 -> EBIC0.50
end
end

function out = fit_liu2026(dat, opts, cuts, opt)
crits = arrayfun(@crit_name, opts.gammaE, 'UniformOutput', false);
C = numel(crits);
out.criteria = crits;
out.support = cell(1, C); out.coef = cell(1, C); out.cif = cell(1, C);
r = cell(1, 2);
for k = 1:2
    st = 2 * ones(size(dat.status));          % dependent censoring: everything else
    st(dat.status == k) = 1;
    r{k} = csm_select(dat.Y, st, dat.Z, cuts, 1, opts.family, opts.tau, opt);
end
if ~strcmp(r{1}.status, 'ok') || ~strcmp(r{2}.status, 'ok')
    bad = r{1}.status; if strcmp(bad, 'ok'), bad = r{2}.status; end
    out.status = repmat({['liu2026_' bad]}, 1, C);
    return
end
out.status = repmat({'ok'}, 1, C);
P2 = csm_prep(dat.Y, dat.status, dat.Z, cuts, 2);   % same standardization and cuts
for c = 1:C
    s1 = r{1}.sel(c); s2 = r{2}.sel(c);
    out.support{c} = [s1.S(1, :); s2.S(1, :); s1.S(2, :)];
    out.coef{c} = [s1.B(1, :); s2.B(1, :); s1.B(2, :)];
    X = [s1.X(1, :); s2.X(1, :); s1.X(2, :)];
    xi = [s1.xi(1, :); s2.xi(1, :); s1.xi(2, :)];
    out.cif{c} = csm_cif(P2, X, xi, dat.Z, opts.grid);
end
end

