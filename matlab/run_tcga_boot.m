function run_tcga_boot(task, varargin)
%RUN_TCGA_BOOT Bootstrap of the TCGA-LUAD sensitivity analysis (Section 6).
%   For every bootstrap sample, Algorithm 1 is run at each working copula in
%   the grid and the EBIC_0.25 support and the cohort-averaged CIFs at 12, 24
%   and 36 months are recorded. The cohort, standardization convention,
%   baseline cutpoints and tuning grids are held fixed. The resampling indices
%   are read from application/data/tcga_boot_idx.csv (433 x 200, 1-based) when it exists,
%   which gives the same samples as the Python run. Otherwise they are drawn
%   from a fixed RandStream, so every array task uses the same 200 samples.
%
%   run_tcga_boot(0)          full data (bootstrap index b = 0)
%   run_tcga_boot(task)       task = 1..ceil(200/chunk), samples (task-1)*chunk+1 ...
%   run_tcga_boot(task, 'chunk', 5, 'families', ["frank"], 'tauGrid', [-0.3 0 0.2 0.5])
%   run_tcga_boot(task, 'cuts', [2.646546339 7.489232995 12 18 30], 'tag', 'p5')   finer partition
%
%   Output: results/tcga_boot/boot_<first b>.csv, one row per (b, family, tau).
%
%   Other cohorts (same columns Y, status, covariates):
%   run_tcga_boot(task, 'data', 'application/hrs/data/hrs_main.csv', ...
%       'cutsfile', 'application/hrs/data/hrs_cuts.csv', 'out', 'hrs_boot', ...
%       'tauGrid', [-0.2 0 0.2 0.4 0.6], 'times', [2 5 10 15], 'chunk', 2)
%   Without an index file the samples come from a fixed RandStream.
ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'chunk', 5);
addParameter(ip, 'families', "frank");
addParameter(ip, 'tauGrid', [-0.3 0 0.2 0.5]);
addParameter(ip, 'times', [12 24 36]);
addParameter(ip, 'gE', 0.25);
addParameter(ip, 'cuts', []);      % internal cutpoints; [] = application/data/tcga_exclude_unknown_cuts.csv
addParameter(ip, 'tag', '');       % output subfolder suffix, e.g. 'p5'
addParameter(ip, 'data', fullfile('application', 'data', 'tcga_exclude_unknown.csv'));
addParameter(ip, 'cutsfile', fullfile('application', 'data', 'tcga_exclude_unknown_cuts.csv'));
addParameter(ip, 'idxfile', fullfile('application', 'data', 'tcga_boot_idx.csv'));
addParameter(ip, 'out', 'tcga_boot');   % results/<out>
addParameter(ip, 'B', 200);
parse(ip, varargin{:});
root = ip.Results.root; chunk = ip.Results.chunk;
fams = string(ip.Results.families); tauGrid = ip.Results.tauGrid;
times = ip.Results.times(:)'; nt = numel(times); gE = ip.Results.gE;

D = readmatrix(fullfile(root, ip.Results.data), 'NumHeaderLines', 1);
if isempty(ip.Results.cuts)
    c = readmatrix(fullfile(root, ip.Results.cutsfile));   % one line, no header
else
    c = ip.Results.cuts;
end
cuts = [0 c(:)' Inf];
n = size(D, 1);
fidx = fullfile(root, ip.Results.idxfile);
if exist(fidx, 'file')
    IDX = readmatrix(fidx);                 % same indices as the Python run
else
    rs = RandStream('mt19937ar', 'Seed', 20261013);
    IDX = randi(rs, n, n, ip.Results.B);    % deterministic, identical in every task
end
Bmax = size(IDX, 2);
if size(IDX, 1) ~= n, error('index file has %d rows, data have %d', size(IDX, 1), n); end
if task == 0
    bs = 0;
else
    bs = (task - 1) * chunk + 1 : min(Bmax, task * chunk);
    if isempty(bs), error('task %d exceeds %d samples', task, Bmax); end
end
K = 2;
opt = csm_options('eta', 0.05, 'gE', [0 0.25 0.5]);
outdir = fullfile(root, 'results', ip.Results.out);
if ~isempty(ip.Results.tag), outdir = [outdir '_' ip.Results.tag]; end
if ~exist(outdir, 'dir'), mkdir(outdir); end

hdr = ["b" "family" "tau" "status" "sup_event1" "sup_event2" "sup_dropout" ...
       compose("F1_t%d", 1:nt) compose("F2_t%d", 1:nt) "seconds"];
rows = strings(0, numel(hdr));
for b = bs
    if b == 0, idx = (1:n)'; else, idx = IDX(:, b); end
    Y = D(idx, 1); st = D(idx, 2); Z = D(idx, 3:end);
    for f = 1:numel(fams)
        for g = 1:numel(tauGrid)
            t0 = tic;
            row = strings(1, numel(hdr));
            row(1:3) = [string(b), fams(f), compose("%.2f", tauGrid(g))];
            try
                res = csm_select(Y, st, Z, cuts, K, char(fams(f)), tauGrid(g), opt);
                row(4) = string(res.status);
                if strcmp(res.status, 'ok')
                    j = find(abs([res.sel.gE] - gE) < 1e-12, 1);
                    s = res.sel(j);
                    for m = 1:K + 1
                        jj = find(s.S(m, :));
                        if isempty(jj), row(4 + m) = ""; else, row(4 + m) = strjoin(string(jj), ";"); end
                    end
                    F = csm_cif(res.P, s.X, s.xi, Z, times);
                    row(8 : 7 + 2*nt) = [compose("%.10g", mean(F{1}, 1)) compose("%.10g", mean(F{2}, 1))];
                end
            catch ME
                row(4) = "error: " + string(regexprep(ME.message, '[,"\r\n]', ' '));
            end
            row(end) = compose("%.2f", toc(t0));
            rows(end + 1, :) = row; %#ok<AGROW>
        end
    end
    fprintf('bootstrap sample %d done\n', b);
end
T = array2table(rows, 'VariableNames', cellstr(hdr));
writetable(T, fullfile(outdir, sprintf('boot_%04d.csv', bs(1))));
fprintf('samples %d-%d written\n', bs(1), bs(end));
end
