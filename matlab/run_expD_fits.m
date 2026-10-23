function run_expD_fits(task, varargin)
%RUN_EXPD_FITS Experiment D: a tau-dependent true signal. Data are generated
%inside the job (expD_gen) for n in {600, 2000, 5000}, 200 replications each,
%and Algorithm 1 is run over tau in {-0.3, -0.2, ..., 0.5}. Output columns are
%the same as run_path_fits.m.
%   run_expD_fits(task)            task 1..80 (10 + 20 + 50 chunks)
ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'cDrop', 0.15);
addParameter(ip, 'tauGrid', -0.3:0.1:0.5);
addParameter(ip, 'times', [1 2 4 6]);
addParameter(ip, 'reps', 200);
addParameter(ip, 'signal', 'strong');   % 'moderate' gives the moderate version (sid prefix Dm, folder expDm)
parse(ip, varargin{:});
root = ip.Results.root; cDrop = ip.Results.cDrop;
tauGrid = round(ip.Results.tauGrid * 10) / 10;
times = ip.Results.times(:)'; nt = numel(times);
ns = [600 2000 5000]; chunks = [20 10 4];
nch = ceil(ip.Results.reps ./ chunks); cs = cumsum(nch);
s = find(task <= cs, 1);
if isempty(s), error('task %d exceeds %d', task, cs(end)); end
prev = [0 cs];
from = (task - prev(s) - 1) * chunks(s) + 1;
to = min(ip.Results.reps, from + chunks(s) - 1);
sig = ip.Results.signal; pre = 'D'; if strcmp(sig, 'moderate'), pre = 'Dm'; end
n = ns(s); sid = sprintf('%s%d', pre, n);
K = 2; p = 10; cuts = [0 1 2 Inf];
gE = [0 0.25 0.5]; crit = ["BIC" "EBIC0.25" "EBIC0.50"];
opt = csm_options('eta', 0.05, 'gE', gE);
outdir = fullfile(root, 'results', ['exp' pre]);
if ~exist(outdir, 'dir'), mkdir(outdir); end

hdr = ["sid" "rep" "tau" "criterion" "status" "sup_event1" "sup_event2" "sup_dropout" ...
       "coef_event1" "coef_event2" "coef_dropout" ...
       compose("F1_t%d", 1:nt) compose("F2_t%d", 1:nt) ...
       compose("T1_t%d", 1:nt) compose("T2_t%d", 1:nt) ...
       compose("PL1_t%d", 1:nt) compose("PU1_t%d", 1:nt) compose("PL2_t%d", 1:nt) compose("PU2_t%d", 1:nt) ...
       "ell_full" "mle_status" "ell_refit" "xi_refit" "seconds"];
rows = strings(0, numel(hdr));
for r = from:to
    dat = expD_gen(n, 20261012 + 100000 * s + r + 7000000 * strcmp(sig, 'moderate'), cDrop, sig);
    [Ft1, Ft2] = sim_true_cif(dat.Z, dat.B, times);
    T1 = mean(Ft1, 1); T2 = mean(Ft2, 1);
    cen = @(t) mean(dat.Y <= t & (dat.status == 3 | dat.status == 0));
    PL1 = arrayfun(@(t) mean(dat.Y <= t & dat.status == 1), times);
    PL2 = arrayfun(@(t) mean(dat.Y <= t & dat.status == 2), times);
    PU1 = PL1 + arrayfun(cen, times); PU2 = PL2 + arrayfun(cen, times);
    pet = [compose("%.10g", PL1) compose("%.10g", PU1) compose("%.10g", PL2) compose("%.10g", PU2)];
    for g = 1:numel(tauGrid)
        t0 = tic;
        try
            res = csm_select(dat.Y, dat.status, dat.Z, cuts, K, 'frank', tauGrid(g), opt);
            st = string(res.status);
        catch ME
            res = []; st = "error: " + string(regexprep(ME.message, '[,"\r\n]', ' '));
        end
        ellFull = NaN; mleSt = "none";
        if st == "ok"
            cop = struct('family', 'frank', 'theta', csm_tau2theta('frank', tauGrid(g)));
            X0 = zeros(K + 1, p);
            mle = csm_fit(res.P, cop, X0, csm_breslow_init(res.P, X0), struct('kind', 'none'), [], opt, true);
            ellFull = mle.ell; mleSt = string(mle.status);
        end
        secs = toc(t0);
        for c = 1:numel(gE)
            row = strings(1, numel(hdr));
            row(1:5) = [string(sid), string(r), compose("%.2f", tauGrid(g)), crit(c), st];
            row(end) = compose("%.2f", secs);
            base = 11 + 2*nt;
            row(base + 1 : base + 2*nt) = [compose("%.10g", T1) compose("%.10g", T2)];
            row(base + 2*nt + 1 : base + 6*nt) = pet;
            row(base + 6*nt + 1) = compose("%.10g", ellFull);
            row(base + 6*nt + 2) = mleSt;
            if st == "ok"
                sel = res.sel(c);
                for m = 1:3
                    idx = find(sel.S(m, :));
                    if isempty(idx), row(5 + m) = ""; else, row(5 + m) = strjoin(string(idx), ";"); end
                    row(8 + m) = strjoin(compose("%.6g", sel.X(m, :)), ";");
                end
                F = csm_cif(res.P, sel.X, sel.xi, dat.Z, times);
                row(12 : 11 + 2*nt) = [compose("%.10g", mean(F{1}, 1)) compose("%.10g", mean(F{2}, 1))];
                row(base + 6*nt + 3) = compose("%.10g", sel.ell);
                row(base + 6*nt + 4) = strjoin(compose("%.8g", sel.xi(:)'), ";");
            end
            rows(end + 1, :) = row; %#ok<AGROW>
        end
    end
    fprintf('%s rep %d done (rates %s)\n', sid, r, mat2str(mean(dat.status == [1 2 3 0], 1), 3));
end
T = array2table(rows, 'VariableNames', cellstr(hdr));
writetable(T, fullfile(outdir, sprintf('expD_%s_%04d.csv', sid, from)));
fprintf('%s reps %d-%d written\n', sid, from, to);
end
