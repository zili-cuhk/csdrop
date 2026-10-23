function run_liu_fits(task, varargin)
%RUN_LIU_FITS Experiment A fits with model-based standard errors, for the
%selection and estimation tables in the format of Liu, Wang, Wang and Song
%(Statistica Sinica 36, 999-1022).
%
%   For each Experiment A dataset four fits are computed, all with Wald
%   standard errors from csm_wald:
%     FULL    unpenalized joint MLE at the true tau with all p covariates
%     PROP    proposed method at the true tau, EBIC_{0.25} support, refitted
%     WI      proposed method at tau = 0 (working independence), EBIC_{0.25}
%     ORACLE  unpenalized joint MLE at the true tau on the true support
%   PROP and WI reuse the supports already selected by run_prop_fits
%   (results/prop/sel_PROP_T{00,02,05}_<sid>_*.csv), so Algorithm 1 is not
%   rerun. The refit on a fixed support is the unique unpenalized MLE, and its
%   coefficients are checked against the stored ones (column maxdiff).
%
%   Scenarios: every cell of Experiment A (A01-A36, n = 400, 600, 1000), the
%   larger-dimension cell H01 and the Study 2 cells S01-S07, in the order of
%   scenarios.csv. For cells with a PROP_TRUE fit (H, S) that fit is the
%   proposed method; otherwise PROP_T<10 tau> is used.
%
%   run_liu_fits(task)                      task = 1 .. 44 * ceil(500 / chunk)
%   run_liu_fits(task, 'chunk', 25, 'crit', 'EBIC0.25')
%
%   Output: results/liu/liu_<sid>_<from>.csv with one row per
%   (rep, fit, margin) and columns sid rep fit margin status support
%   coef (p values, raw scale, ';'-separated) se (same) maxdiff seconds.
ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'chunk', 25);
addParameter(ip, 'crit', 'EBIC0.25');
addParameter(ip, 'fits', {'FULL', 'PROP', 'WI', 'ORACLE'});
parse(ip, varargin{:});
root = ip.Results.root; chunk = ip.Results.chunk; crit = string(ip.Results.crit);
fits = string(ip.Results.fits);

S = read_text_table(fullfile(root, 'scenarios.csv'));
S = S(ismember(S.exp, ["A" "H" "S"]), :);
Rn = str2double(S.R);
nch = ceil(Rn / chunk); cs = cumsum(nch);
s = find(task <= cs, 1);
if isempty(s), error('task must be in 1..%d', cs(end)); end
prev = [0; cs(:)];
from = (task - prev(s) - 1) * chunk + 1;
to = min(Rn(s), from + chunk - 1);
sid = S.sid(s); signal = S.signal(s); p = str2double(S.p(s));
tau0 = str2double(S.tau(s));
baseline = "pc";
if any(strcmp(S.Properties.VariableNames, 'baseline')), baseline = S.baseline(s); end
Btrue = sim_coefs(signal, p);
K = 2;
opt = csm_options();
nopen = struct('kind', 'none');
margins = ["event1" "event2" "dropout"];

if ~isempty(dir(fullfile(root, 'results', 'prop', sprintf('sel_PROP_TRUE_%s_*.csv', sid))))
    fitTrue = 'PROP_TRUE';
else
    fitTrue = sprintf('PROP_T%02d', round(10 * tau0));
end
selProp = read_sel(root, fitTrue, sid, crit);
selWI = read_sel(root, 'PROP_T00', sid, crit);

outdir = fullfile(root, 'results', 'liu');
if ~exist(outdir, 'dir'), mkdir(outdir); end
rows = strings(0, 10);
for r = from:to
    dat = sim_read_dataset(root, sid, r);
    cuts = sim_cuts("auto", baseline, dat.Y, dat.status);
    P = csm_prep(dat.Y, dat.status, dat.Z, cuts, K);
    for f = fits
        t0 = tic;
        stored = [];
        switch f
            case 'FULL'
                tau = tau0; Sup = true(K + 1, p);
            case 'ORACLE'
                tau = tau0; Sup = Btrue ~= 0;
            case 'PROP'
                tau = tau0; [Sup, stored] = lookup(selProp, r, p);
            case 'WI'
                tau = 0; [Sup, stored] = lookup(selWI, r, p);
        end
        cop = struct('family', 'frank', 'theta', csm_tau2theta('frank', tau));
        st = "ok"; B = nan(K + 1, p); se = nan(K + 1, p); md = NaN;
        if isempty(Sup)
            st = "missing_selection";
        else
            try
                [rf, st] = refit(P, cop, Sup, f, nopen, opt, K, p);
                if st == "ok"
                    B = rf.X ./ P.sd;
                    [se, okw] = csm_wald(P, cop, rf.X, rf.xi, Sup);
                    if ~okw, st = "information_not_pd"; end
                    if ~isempty(stored), md = max(abs(B(:) - stored(:))); end
                end
            catch ME
                st = "error: " + string(regexprep(ME.message, '[,"\r\n]', ' '));
            end
        end
        secs = toc(t0);
        if isempty(Sup), Sup = false(K + 1, p); end    % no selection stored for this rep
        for m = 1:K + 1
            idx = find(Sup(m, :));
            if isempty(idx), sup = ""; else, sup = strjoin(string(idx), ";"); end
            rows(end + 1, :) = [sid, string(r), f, margins(m), st, sup, ...
                strjoin(compose("%.10g", B(m, :)), ";"), ...
                strjoin(compose("%.10g", se(m, :)), ";"), ...
                compose("%.3g", md), compose("%.2f", secs)]; %#ok<AGROW>
        end
    end
end
T = array2table(rows, 'VariableNames', ...
    {'sid', 'rep', 'fit', 'margin', 'status', 'support', 'coef', 'se', 'maxdiff', 'seconds'});
writetable(T, fullfile(outdir, sprintf('liu_%s_%04d.csv', sid, from)));
fprintf('%s reps %d-%d written\n', sid, from, to);
end

function [rf, st] = refit(P, cop, Sup, f, nopen, opt, K, p)
% Unpenalized MLE on the support Sup. FULL uses the zero, marginal and
% weak-ridge seeds of csm_thr_mle and keeps the admissible fit with the
% smallest objective; the other fits start from the independence fit on the
% same support.
indep = struct('family', 'indep', 'theta', 0);
X0 = zeros(K + 1, p);
fi = csm_fit(P, indep, X0, csm_breslow_init(P, X0), nopen, Sup, opt, true);
if f == "FULL"
    fr = csm_fit(P, indep, X0, csm_breslow_init(P, X0), ...
        struct('kind', 'ridge', 'lam', opt.ridge_lam), [], opt, true);
    seeds = {X0, fi.X, fr.X};
else
    seeds = {fi.X, X0};
end
rf = [];
for k = 1:numel(seeds)
    g = csm_fit(P, cop, seeds{k}, csm_breslow_init(P, seeds{k}), nopen, Sup, opt, true);
    if strcmp(g.status, 'ok') && (isempty(rf) || g.Q < rf.Q), rf = g; end
end
if isempty(rf), st = "no_admissible_refit"; else, st = "ok"; end
end

function sel = read_sel(root, fit_id, sid, crit)
% Stored supports and coefficients of run_prop_fits for one scenario.
F = dir(fullfile(root, 'results', 'prop', sprintf('sel_%s_%s_*.csv', fit_id, sid)));
if isempty(F), error('no stored selections for %s %s', fit_id, sid); end
sel = table();
for k = 1:numel(F)
    T = read_text_table(fullfile(F(k).folder, F(k).name));
    sel = [sel; T(T.criterion == crit & T.status == "ok", :)]; %#ok<AGROW>
end
sel.rep = str2double(sel.rep);
end

function [Sup, B] = lookup(sel, r, p)
Sup = []; B = [];
ms = ["event1" "event2" "dropout"];
T = sel(sel.rep == r, :);
if height(T) ~= 3, return, end
Sup = false(3, p); B = zeros(3, p);
for m = 1:3
    t = T(T.margin == ms(m), :);
    if height(t) ~= 1, Sup = []; B = []; return, end
    if strlength(t.support) > 0
        Sup(m, str2double(split(t.support, ";"))) = true;
    end
    if strlength(t.coef) > 0
        B(m, :) = str2double(split(t.coef, ";"))';
    end
end
end

function T = read_text_table(f)
o = detectImportOptions(f, 'Delimiter', ',');
o = setvartype(o, 'string');
T = readtable(f, o);
end
