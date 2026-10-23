function run_pathB_fits(task, varargin)
%RUN_PATHB_FITS Exp C-B: Algorithm 1 over a grid of (copula family, Kendall's tau)
%on the saved Experiment B datasets (Clayton/Gumbel truth at tau = 0.5, and Frank
%tau = 0.2 with event-type-dependent dropout, delta = 0.5 or 1). Families: Frank at
%tau = 0, 0.1, ..., 0.5 and Clayton/Gumbel at tau = 0.1, ..., 0.5 (tau = 0 is the
%independence copula for every family). Same output columns as run_path_fits plus
%'family'. For every dataset and grid value it
%records the thresholded supports and the cohort-averaged CIFs of the selected
%refits for BIC, EBIC0.25 and EBIC0.50, together with the cohort-averaged true
%CIFs. Robust/possible supports and CIF bounds are formed afterwards for any
%subset of the grid.
%
%   run_pathB_fits(task)            task 1..400 (8 scenarios x 50 chunks of 10)
%   Output: results/pathB/pathB_<sid>_<from>.csv, one row per (rep, family, tau, criterion).

ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'chunk', 10);
addParameter(ip, 'tauGrid', 0:0.1:0.5);
addParameter(ip, 'times', [1 2 4 6]);
parse(ip, varargin{:});
root = ip.Results.root;
chunk = ip.Results.chunk;
tauGrid = ip.Results.tauGrid;
times = ip.Results.times(:)';
nt = numel(times);

S = read_text_table(fullfile(root, 'scenarios.csv'));
S = S(S.exp == "B", :);
Rn = str2double(S.R);
[s, from, to] = task_map(task, Rn, chunk);
sid = S.sid(s);
signal = S.signal(s);
p = str2double(S.p(s));
B = sim_coefs(signal, p);
K = 2;
cuts = [0 1 2 Inf];
gE = [0 0.25 0.5];
crit = ["BIC" "EBIC0.25" "EBIC0.50"];
opt = csm_options('eta', 0.05, 'gE', gE);

outdir = fullfile(root, 'results', 'pathB');
fams = [repmat("frank", 1, numel(tauGrid)), repmat("clayton", 1, numel(tauGrid) - 1), repmat("gumbel", 1, numel(tauGrid) - 1)];
taus = [tauGrid, tauGrid(2:end), tauGrid(2:end)];
if ~exist(outdir, 'dir'), mkdir(outdir); end

hdr = ["sid" "rep" "tau" "criterion" "status" "sup_event1" "sup_event2" "sup_dropout" ...
       "coef_event1" "coef_event2" "coef_dropout" ...
       compose("F1_t%d", 1:nt) compose("F2_t%d", 1:nt) ...
       compose("T1_t%d", 1:nt) compose("T2_t%d", 1:nt) ...
       compose("PL1_t%d", 1:nt) compose("PU1_t%d", 1:nt) compose("PL2_t%d", 1:nt) compose("PU2_t%d", 1:nt) ...
       "ell_full" "mle_status" "ell_refit" "xi_refit" "family" "seconds"];
rows = strings(0, numel(hdr));

for r = from:to
    dat = sim_read_dataset(root, sid, r);
    [Ft1, Ft2] = sim_true_cif(dat.Z, B, times);
    T1 = mean(Ft1, 1); T2 = mean(Ft2, 1);
    % Assumption-free (Peterson-type) bounds on the cohort CIFs: every subject
    % censored before t either never fails by t (lower) or fails from cause k (upper).
    cen = @(t) mean(dat.Y <= t & (dat.status == 3 | dat.status == 0));
    PL1 = arrayfun(@(t) mean(dat.Y <= t & dat.status == 1), times);
    PL2 = arrayfun(@(t) mean(dat.Y <= t & dat.status == 2), times);
    PU1 = PL1 + arrayfun(cen, times); PU2 = PL2 + arrayfun(cen, times);
    pet = [compose("%.10g", PL1) compose("%.10g", PU1) compose("%.10g", PL2) compose("%.10g", PU2)];
    for g = 1:numel(taus)
        t0 = tic;
        try
            res = csm_select(dat.Y, dat.status, dat.Z, cuts, K, char(fams(g)), taus(g), opt);
            st = string(res.status);
        catch ME
            res = []; st = "error: " + string(regexprep(ME.message, '[,"\r\n]', ' '));
        end
        ellFull = NaN; mleSt = "none";
        if st == "ok"
            % full-model unpenalized fit at this tau: its log-likelihood over the grid is the
            % profile likelihood used by the estimate-tau-then-select comparator
            cop = struct('family', char(fams(g)), 'theta', csm_tau2theta(char(fams(g)), taus(g)));
            X0 = zeros(K + 1, p);
            mle = csm_fit(res.P, cop, X0, csm_breslow_init(res.P, X0), struct('kind', 'none'), [], opt, true);
            ellFull = mle.ell; mleSt = string(mle.status);
        end
        secs = toc(t0);
        for c = 1:numel(gE)
            row = strings(1, numel(hdr));
            row(1:5) = [sid, string(r), compose("%.2f", taus(g)), crit(c), st];
            row(end - 1) = fams(g);
            row(end) = compose("%.2f", secs);
            base = 11 + 2*nt;                                   % last F column
            row(base + 1 : base + 2*nt) = [compose("%.10g", T1) compose("%.10g", T2)];
            row(base + 2*nt + 1 : base + 6*nt) = pet;
            row(base + 6*nt + 1) = compose("%.10g", ellFull);
            row(base + 6*nt + 2) = mleSt;
            if st == "ok"
                sel = res.sel(c);
                for m = 1:3
                    idx = find(sel.S(m, :));
                    if isempty(idx), row(5 + m) = ""; else, row(5 + m) = strjoin(string(idx), ";"); end
                    row(8 + m) = strjoin(compose("%.6g", sel.X(m, :)), ";");   % standardized scale
                end
                F = csm_cif(res.P, sel.X, sel.xi, dat.Z, times);
                row(12 : 11 + 2*nt) = [compose("%.10g", mean(F{1}, 1)) compose("%.10g", mean(F{2}, 1))];
                row(base + 6*nt + 3) = compose("%.10g", sel.ell);
                row(base + 6*nt + 4) = strjoin(compose("%.8g", sel.xi(:)'), ";");   % column-major (K+1) x M
            end
            rows(end + 1, :) = row; %#ok<AGROW>
        end
    end
    fprintf('%s rep %d done\n', sid, r);
end
T = array2table(rows, 'VariableNames', cellstr(hdr));
writetable(T, fullfile(outdir, sprintf('pathB_%s_%04d.csv', sid, from)));
fprintf('%s reps %d-%d written\n', sid, from, to);
end

function T = read_text_table(f)
o = detectImportOptions(f, 'Delimiter', ',');
o = setvartype(o, 'string');
T = readtable(f, o);
end

function [s, from, to] = task_map(task, Rn, chunk)
nch = ceil(Rn / chunk);
cs = cumsum(nch);
s = find(task <= cs, 1);
if isempty(s), error('task index %d exceeds %d', task, cs(end)); end
prev = [0; cs(:)];
c0 = task - prev(s) - 1;
from = c0 * chunk + 1;
to = min(Rn(s), from + chunk - 1);
end
