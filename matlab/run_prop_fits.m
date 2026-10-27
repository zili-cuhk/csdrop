function run_prop_fits(task, fitfun, varargin)
%RUN_PROP_FITS Fit the proposed method (MATLAB fits in fit_plan.csv) on one
%chunk of saved datasets and write results in the common schema.
%
%   run_prop_fits(task, @fit_prop_adapter)
%   run_prop_fits(task, @fit_prop_adapter, 'root', '.', 'chunk', 50, 'fits', {'PROP_T00'})
%   run_prop_fits(task, @fit_prop_adapter, 'exp', 'P', 'chunk', 10)
%       (task index within the scenarios of experiment P only)
%
%   TASK is the LSF array index; the task-to-(scenario, reps) rule is the same
%   as task_map() in R/dgp.R. FITFUN is a handle with signature
%       out = fitfun(dat, opts)
%   See fit_prop_adapter.m for the input and output contract.

ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'chunk', 50);
addParameter(ip, 'fits', {});
addParameter(ip, 'exp', '');
parse(ip, varargin{:});
root = ip.Results.root;
chunk = ip.Results.chunk;

S = read_text_table(fullfile(root, 'scenarios.csv'));
if ~isempty(ip.Results.exp)
    S = S(ismember(S.exp, string(ip.Results.exp)), :);
end
plan = read_text_table(fullfile(root, 'fit_plan.csv'));
Rn = str2double(S.R);
[s, from, to] = task_map(task, Rn, chunk);
sid = S.sid(s);
expname = S.exp(s);
signal = S.signal(s);
p = str2double(S.p(s));
true_family = S.family(s);
true_tau = str2double(S.tau(s));
delta = str2double(S.delta(s));
f2 = 1; baseline = "pc";                         % Study 2 columns (absent before the revision)
if any(strcmp(S.Properties.VariableNames, 'f2')), f2 = str2double(S.f2(s)); end
if any(strcmp(S.Properties.VariableNames, 'baseline')), baseline = S.baseline(s); end

rows = plan.exp == expname & plan.engine == "matlab";
if delta ~= 0
    rows = rows & plan.skip_if_delta == "0";
end
if ~isempty(ip.Results.fits)
    rows = rows & ismember(plan.fit_id, string(ip.Results.fits));
end
if any(strcmp(plan.Properties.VariableNames, 'only_if'))
    onlyIf = plan.only_if; onlyIf(ismissing(onlyIf)) = "";
    rows = rows & (onlyIf == "" | onlyIf == baseline);
end
myplan = plan(rows, :);

B = sim_coefs(signal, p);
grid = 0:0.1:6;
outdir = fullfile(root, 'results', 'prop');
if ~exist(outdir, 'dir'), mkdir(outdir); end

for f = 1:height(myplan)
    fit_id = myplan.fit_id(f);
    opts = struct();
    opts.family = char(resolve(myplan.family_fit(f), true_family));
    opts.tau = str2double(resolve(myplan.tau_fit(f), string(true_tau)));
    opts.penalty = char(myplan.penalty(f));          % 'gselo' or 'none'
    opts.threshold = 0.05;
    if any(strcmp(plan.Properties.VariableNames, 'threshold'))
        thr = str2double(myplan.threshold(f));      % missing / "NA" -> NaN
        if ~isnan(thr), opts.threshold = thr; end
    end
    opts.gammaE = [0 0.25 0.5];
    opts.grid = grid;
    lump = myplan.lump_admin(f) == "1";
    cutSpec = "auto";
    if any(strcmp(plan.Properties.VariableNames, 'cuts')) && ~ismissing(myplan.cuts(f))
        cutSpec = myplan.cuts(f);
    end

    sel = {};   % rows: sid rep fit_id criterion margin support coef status seconds
    cif = {};   % rows: sid rep fit_id criterion cause ise
    for r = from:to
        dat = sim_read_dataset(root, sid, r);
        [Ft1, Ft2] = sim_true_cif(dat.Z, B, grid, f2, baseline);   % truth from the unmodified data
        opts.cuts = sim_cuts(cutSpec, baseline, dat.Y, dat.status);
        if lump
            dat.status(dat.status == 0) = 3;         % administrative -> dropout
        end
        t0 = tic;
        try
            out = fitfun(dat, opts);
        catch ME
            msg = regexprep(ME.message, '[,"\r\n]', ' ');   % keep the CSV well formed
            sel(end+1, :) = {sid, r, fit_id, "ALL", "ALL", "", "", ...
                "error: " + string(msg), toc(t0)}; %#ok<AGROW>
            continue
        end
        secs = toc(t0);
        margins = ["event1" "event2" "dropout"];
        for c = 1:numel(out.criteria)
            crit = string(out.criteria{c});
            st = string(regexprep(char(out.status{c}), '[,"\r\n]', ' '));
            for m = 1:3
                if st == "ok"
                    idx = find(out.support{c}(m, :));
                    if isempty(idx)
                        sup = "";
                    else
                        sup = strjoin(string(idx), ";");
                    end
                    cf = strjoin(compose("%.10g", out.coef{c}(m, :)), ";");
                else
                    sup = ""; cf = "";
                end
                sel(end+1, :) = {sid, r, fit_id, crit, margins(m), sup, cf, st, secs}; %#ok<AGROW>
            end
            if st == "ok"
                cif(end+1, :) = {sid, r, fit_id, crit, 1, ...
                    sim_cif_ise(out.cif{c}{1}, Ft1, grid)}; %#ok<AGROW>
                cif(end+1, :) = {sid, r, fit_id, crit, 2, ...
                    sim_cif_ise(out.cif{c}{2}, Ft2, grid)}; %#ok<AGROW>
            end
        end
    end

    tag = sprintf('%s_%s_%04d.csv', fit_id, sid, from);
    if ~isempty(sel)
        T = cell2table(sel, 'VariableNames', ...
            {'sid', 'rep', 'fit_id', 'criterion', 'margin', 'support', 'coef', 'status', 'seconds'});
        writetable(T, fullfile(outdir, ['sel_' tag]));
    end
    if ~isempty(cif)
        T = cell2table(cif, 'VariableNames', {'sid', 'rep', 'fit_id', 'criterion', 'cause', 'ise'});
        writetable(T, fullfile(outdir, ['cif_' tag]));
    end
    fprintf('%s %s reps %d-%d done\n', sid, fit_id, from, to);
end
end

function T = read_text_table(f)
% Read every column as string so "true", "NA" and numbers survive unchanged.
o = detectImportOptions(f, 'Delimiter', ',');
o = setvartype(o, 'string');
T = readtable(f, o);
end

function v = resolve(spec, truth)
% "true" in fit_plan.csv means "use the scenario's true value".
if spec == "true"
    v = string(truth);
else
    v = string(spec);
end
end

function [s, from, to] = task_map(task, Rn, chunk)
% Same rule as task_map() in R/dgp.R.
nch = ceil(Rn / chunk);
cs = cumsum(nch);
s = find(task <= cs, 1);
if isempty(s), error('task index %d exceeds %d', task, cs(end)); end
prev = [0; cs(:)];
c0 = task - prev(s) - 1;
from = c0 * chunk + 1;
to = min(Rn(s), from + chunk - 1);
end
