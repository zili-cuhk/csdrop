function run_expF_fits(task, varargin)
%RUN_EXPF_FITS Experiment F: coverage of the Imbens-Manski intervals for the
%cohort-averaged CIF bounds when the standard errors come from the bootstrap
%(Theorem 4(iv) and Section 3.6 of the main text).
%
%   For one Experiment A dataset, Algorithm 1 is run at every grid value on the
%   full data and on B bootstrap samples of subjects (tuning grids, partition
%   and threshold fixed). The output contains, for each cause and time, the
%   full-data CIFs at every grid value, the bootstrap standard errors of the
%   lower and upper bounds and of each grid value, and the true cohort-averaged
%   CIF of the sample. Coverage is computed afterwards by agg_expF.py.
%
%   run_expF_fits(task)                    task = 1 .. 200 * numel(sids)
%   run_expF_fits(task, 'B', 100, 'tauGrid', [0 0.25 0.5], 'reps', 200)
%
%   Scenarios: Experiment A with n = 600, p = 10, both signal levels and
%   tau0 in {0.2, 0.5}, taken from scenarios.csv. Output:
%   results/expF/expF_<sid>_<rep>.csv, one row per (cause, time).
%
%   Options for the follow-up experiments of the revision
%     'tag'   output folder and file prefix (default 'expF')
%     'n'     sample sizes of the Experiment A cells used (default 600)
%     'taus'  true tau0 values of the cells used (default [0.2 0.5])
%     'B'     0 skips the bootstrap (standard-error columns are NaN)
%   Experiment G (near ties, Theorem 4(v)):
%     run_expF_fits(task, 'tag', 'expG', 'tauGrid', [0.4 0.45 0.5], 'taus', 0.5)
%   Experiment H (grid refinement, Theorem 5):
%     run_expF_fits(task, 'tag', 'expH', 'B', 0, 'tauGrid', 0:0.025:0.5, 'n', [600 1000])
ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'B', 100);
addParameter(ip, 'tauGrid', [0 0.25 0.5]);
addParameter(ip, 'times', [1 2 4 6]);
addParameter(ip, 'reps', 200);
addParameter(ip, 'gE', 0.25);
addParameter(ip, 'tag', 'expF');
addParameter(ip, 'n', 600);
addParameter(ip, 'taus', [0.2 0.5]);
parse(ip, varargin{:});
tag = char(ip.Results.tag);
root = ip.Results.root; Bn = ip.Results.B; tauGrid = ip.Results.tauGrid;
times = ip.Results.times(:)'; nt = numel(times); G = numel(tauGrid); gE = ip.Results.gE;

S = read_text_table(fullfile(root, 'scenarios.csv'));
S = S(S.exp == "A" & ismember(str2double(S.n), ip.Results.n) & str2double(S.p) == 10 & ...
      ismember(round(str2double(S.tau) * 100), round(100 * ip.Results.taus)), :);
nsc = height(S); R = ip.Results.reps;
if task < 1 || task > nsc * R, error('task must be in 1..%d', nsc * R); end
s = ceil(task / R); rep = task - (s - 1) * R;
sid = S.sid(s); signal = S.signal(s); p = str2double(S.p(s));
Btrue = sim_coefs(signal, p);
K = 2; cuts = [0 1 2 Inf];
opt = csm_options('eta', 0.05, 'gE', [0 0.25 0.5]);
outdir = fullfile(root, 'results', tag);
if ~exist(outdir, 'dir'), mkdir(outdir); end

dat = sim_read_dataset(root, sid, rep);
n = numel(dat.Y);
[Ft1, Ft2] = sim_true_cif(dat.Z, Btrue, times);
Ftrue = [mean(Ft1, 1); mean(Ft2, 1)];                    % K x nt

Ffull = grid_cifs(dat.Y, dat.status, dat.Z, (1:n)');      % K x nt x G
if any(isnan(Ffull(:))), error('full-data fit failed for %s rep %d', sid, rep); end

rs = RandStream('mt19937ar', 'Seed', 20261020 + 1000 * s + rep);
Fb = nan(K, nt, G, Bn);
t0 = tic;
for b = 1:Bn
    idx = randi(rs, n, n, 1);
    Fb(:, :, :, b) = grid_cifs(dat.Y, dat.status, dat.Z, idx);
end
if Bn > 0
    ok = reshape(all(all(all(~isnan(Fb), 1), 2), 3), [], 1);
    Fb = Fb(:, :, :, ok); nok = sum(ok);
    if nok < 10, error('only %d admissible bootstrap samples', nok); end
    Lb = reshape(min(Fb, [], 3), K, nt, nok); Ub = reshape(max(Fb, [], 3), K, nt, nok);
    seL = std(Lb, 0, 3); seU = std(Ub, 0, 3); seG = std(Fb, 0, 4);
else                                                      % no bootstrap
    nok = 0; seL = nan(K, nt); seU = nan(K, nt); seG = nan(K, nt, G);
end
[Lf, gL] = min(Ffull, [], 3); [Uf, gU] = max(Ffull, [], 3);

hdr = ["sid" "rep" "cause" "t" compose("F_g%d", 1:G) "L" "U" "gL" "gU" "seL" "seU" ...
       compose("se_g%d", 1:G) "truth" "B_ok" "seconds"];
rows = strings(0, numel(hdr));
for k = 1:K
    for j = 1:nt
        rows(end + 1, :) = [sid, string(rep), string(k), string(times(j)), ...
            compose("%.10g", squeeze(Ffull(k, j, :))'), compose("%.10g", [Lf(k, j) Uf(k, j)]), ...
            string(gL(k, j)), string(gU(k, j)), compose("%.10g", [seL(k, j) seU(k, j)]), ...
            compose("%.10g", squeeze(seG(k, j, :))'), compose("%.10g", Ftrue(k, j)), ...
            string(nok), compose("%.1f", toc(t0))]; %#ok<AGROW>
    end
end
T = array2table(rows, 'VariableNames', cellstr(hdr));
writetable(T, fullfile(outdir, sprintf('%s_%s_%04d.csv', tag, sid, rep)));
fprintf('%s rep %d written (%d of %d bootstrap samples admissible)\n', sid, rep, nok, Bn);

    function F = grid_cifs(Y, st, Z, idx)
        % cohort-averaged CIFs of the EBIC-selected refits at every grid value
        F = nan(K, nt, G);
        Yb = Y(idx); sb = st(idx); Zb = Z(idx, :);
        for g = 1:G
            try
                res = csm_select(Yb, sb, Zb, cuts, K, 'frank', tauGrid(g), opt);
            catch
                F(:) = NaN; return
            end
            if ~strcmp(res.status, 'ok'), F(:) = NaN; return; end
            jj = find(abs([res.sel.gE] - gE) < 1e-12, 1);
            sel = res.sel(jj);
            Fc = csm_cif(res.P, sel.X, sel.xi, Zb, times);
            F(1, :, g) = mean(Fc{1}, 1); F(2, :, g) = mean(Fc{2}, 1);
        end
    end
end

function T = read_text_table(f)
o = detectImportOptions(f, 'Delimiter', ',');
o = setvartype(o, 'string');
T = readtable(f, o);
end
