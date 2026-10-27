function S = agg_tcga_boot(varargin)
%AGG_TCGA_BOOT Summaries of run_tcga_boot for Section 6 (Tables 4 and 5).
%   S = agg_tcga_boot()                       Frank grid only
%   S = agg_tcga_boot('dir', 'results/hrs_boot', 'data', 'application/hrs/data/hrs_main.csv', ...
%       'margins', ["Dementia" "Death without dementia" "Dropout"])   another cohort
%   S = agg_tcga_boot('family', "all")        intersection/union over all
%                                             (family, tau) pairs that were run
%   Reads results/tcga_boot/boot_*.csv (b = 0 is the full data) and writes
%   results/tcga_boot/summary_freq_<family>.csv and summary_cif_<family>.csv.
%
%   Selection frequency of each (margin, covariate) at each working copula,
%   frequency of membership in the bootstrap robust and possible supports,
%   frequency with which a bootstrap support equals the full-data support,
%   bootstrap standard errors of the CIF bounds, Imbens-Manski intervals
%   (Imbens and Manski, 2004; Stoye, 2009), and the envelope interval of
%   Theorem 4(v) (min/max of the pointwise 95% limits over the grid).
%   Both intervals are also given with the normalized interquartile range
%   (q75 - q25) / 1.349 of the bootstrap replicates as the standard error,
%   which is consistent without uniform integrability of the bootstrap
%   replicates (Web Appendix S10).
%   Output columns of summary_cif_<family>.csv: cause time L U seL seU IM_lo
%   IM_hi env_lo env_hi seL_iqr seU_iqr IMiqr_lo IMiqr_hi enviqr_lo enviqr_hi B_ok.
ip = inputParser;
addParameter(ip, 'root', '.');
addParameter(ip, 'family', "frank");
addParameter(ip, 'alpha', 0.05);
addParameter(ip, 'tag', '');
addParameter(ip, 'dir', '');       % results folder, e.g. 'results/hrs_boot'
addParameter(ip, 'data', '');      % cohort csv whose header names the covariates
addParameter(ip, 'margins', ["Recorded PFS event" "Death without earlier PFS" "Dropout"]);
parse(ip, varargin{:});
root = ip.Results.root; famSel = string(ip.Results.family); alpha = ip.Results.alpha;
dirr = fullfile(root, 'results', 'tcga_boot');
if ~isempty(ip.Results.tag), dirr = [dirr '_' ip.Results.tag]; end
if ~isempty(ip.Results.dir), dirr = fullfile(root, ip.Results.dir); end
fl = dir(fullfile(dirr, 'boot_*.csv'));
if isempty(fl), error('no files in %s', dirr); end
T = [];
for i = 1:numel(fl)
    o = detectImportOptions(fullfile(dirr, fl(i).name), 'Delimiter', ',');
    o = setvartype(o, 'string');
    T = [T; readtable(fullfile(dirr, fl(i).name), o)]; %#ok<AGROW>
end
if famSel ~= "all", T = T(T.family == famSel, :); end
if isempty(T), error('no rows for family %s', famSel); end

names = ["Age per 10 years" "Male sex" "Stage II" "Stage III-IV" "Node-positive" "T3-T4"];
if ~isempty(ip.Results.data)
    fid = fopen(fullfile(root, ip.Results.data)); h = strsplit(strtrim(fgetl(fid)), ','); fclose(fid);
    names = string(strrep(h(3:end), '"', ''));
end
margins = string(ip.Results.margins);
p = numel(names); D1 = 3; K = 2;
fc = startsWith(T.Properties.VariableNames, 'F1_');
nt = sum(fc);
spec = unique(T.family + "|" + T.tau, 'stable');
G = numel(spec);
bAll = unique(str2double(T.b));
if ~any(bAll == 0), error('full-data run (task 0) not found'); end
bBoot = bAll(bAll > 0)';

% ---- read one (b, spec) row into support and CIFs
    function [ok, Sm, F] = getrow(b, g)
        r = find(str2double(T.b) == b & T.family + "|" + T.tau == spec(g), 1);
        ok = ~isempty(r) && T.status(r) == "ok";
        Sm = false(D1, p); F = nan(K, nt);
        if ~ok, return; end
        cols = ["sup_event1" "sup_event2" "sup_dropout"];
        for m = 1:D1
            v = T.(cols(m))(r);
            if ~ismissing(v) && strlength(v) > 0
                Sm(m, str2double(split(v, ";"))) = true;
            end
        end
        for k = 1:K
            F(k, :) = str2double(T{r, cellstr("F" + k + "_t" + (1:nt))});
        end
    end

% ---- full data
Sfull = false(D1, p, G); Ffull = nan(K, nt, G);
for g = 1:G
    [ok, Sfull(:, :, g), Ffull(:, :, g)] = getrow(0, g);
    if ~ok, error('full-data fit failed at %s', spec(g)); end
end
SRfull = all(Sfull, 3); SPfull = any(Sfull, 3);
[Lf, gLf] = min(Ffull, [], 3); [Uf, gUf] = max(Ffull, [], 3);

% ---- bootstrap
nb = numel(bBoot);
Sb = false(D1, p, G, nb); Fb = nan(K, nt, G, nb); okb = false(nb, 1);
for i = 1:nb
    oks = true;
    for g = 1:G
        [ok, Sb(:, :, g, i), Fb(:, :, g, i)] = getrow(bBoot(i), g);
        oks = oks && ok;
    end
    okb(i) = oks;
end
Sb = Sb(:, :, :, okb); Fb = Fb(:, :, :, okb); nok = sum(okb);
freq = mean(Sb, 4);                                   % D1 x p x G
SRb = all(Sb, 3); SPb = any(Sb, 3);                   % D1 x p x 1 x nok
freqR = mean(SRb, 4); freqP = mean(SPb, 4);
eqG = zeros(1, G);
for g = 1:G
    eqG(g) = mean(all(all(Sb(:, :, g, :) == Sfull(:, :, g), 1), 2));
end
eqR = mean(all(all(SRb == SRfull, 1), 2)); eqP = mean(all(all(SPb == SPfull, 1), 2));
[Lb, gLb] = min(Fb, [], 3); [Ub, gUb] = max(Fb, [], 3);       % K x nt x 1 x nok
Lb = squeeze4(Lb); Ub = squeeze4(Ub);
sameArg = min(mean(squeeze4(gLb) == gLf, 3), mean(squeeze4(gUb) == gUf, 3));
seL = std(Lb, 0, 3); seU = std(Ub, 0, 3); seG = std(Fb, 0, 4);  % seG: K x nt x G
qL = iqrse(Lb, 3); qU = iqrse(Ub, 3); qG = iqrse(Fb, 4);
[IMlo, IMhi, ENVlo, ENVhi] = intervals(Lf, Uf, Ffull, seL, seU, seG, alpha);
[IQlo, IQhi, EQlo, EQhi] = intervals(Lf, Uf, Ffull, qL, qU, qG, alpha);

% ---- print
fprintf('Family %s: %d bootstrap samples read, %d with admissible fits at all %d grid values\n', famSel, nb, nok, G);
fprintf('Full data: robust = possible: %d\n', isequal(SRfull, SPfull));
fprintf('Bootstrap support equals full-data support: per grid value %s, robust %.3f, possible %.3f\n', ...
    mat2str(eqG, 3), eqR, eqP);
fprintf('%-28s %-18s %s  robust possible\n', 'Margin', 'Covariate', strjoin(spec', ' '));
for m = 1:D1
    for j = 1:p
        fprintf('%-28s %-18s %s  %.2f %.2f\n', margins(m), names(j), ...
            sprintf('%.2f ', squeeze(freq(m, j, :))), freqR(m, j), freqP(m, j));
    end
end
fprintf('Share of samples with the same argmin and argmax as the full data: %.3f\n', min(sameArg(:)));
for k = 1:K
    for t = 1:nt
        fprintf('cause %d t%d: bounds [%.3f, %.3f]  se (%.4f, %.4f)  IM [%.3f, %.3f]  envelope [%.3f, %.3f]\n', ...
            k, t, Lf(k, t), Uf(k, t), seL(k, t), seU(k, t), IMlo(k, t), IMhi(k, t), ENVlo(k, t), ENVhi(k, t));
        fprintf('             IQR se (%.4f, %.4f)  IM [%.3f, %.3f]  envelope [%.3f, %.3f]\n', ...
            qL(k, t), qU(k, t), IQlo(k, t), IQhi(k, t), EQlo(k, t), EQhi(k, t));
    end
end

% ---- write
R = strings(0, 4 + G);
for m = 1:D1
    for j = 1:p
        R(end + 1, :) = [margins(m), names(j), compose("%.3f", squeeze(freq(m, j, :))'), ...
            compose("%.3f", freqR(m, j)), compose("%.3f", freqP(m, j))]; %#ok<AGROW>
    end
end
TF = array2table(R, 'VariableNames', [{'margin', 'covariate'}, cellstr("sel_" + matlab.lang.makeValidName(spec')), {'robust', 'possible'}]);
writetable(TF, fullfile(dirr, sprintf('summary_freq_%s.csv', famSel)));
R = strings(0, 17);
for k = 1:K
    for t = 1:nt
        R(end + 1, :) = [string(k), string(t), compose("%.4f", [Lf(k, t) Uf(k, t) seL(k, t) seU(k, t) ...
            IMlo(k, t) IMhi(k, t) ENVlo(k, t) ENVhi(k, t) qL(k, t) qU(k, t) ...
            IQlo(k, t) IQhi(k, t) EQlo(k, t) EQhi(k, t)]), string(nok)]; %#ok<AGROW>
    end
end
TC = array2table(R, 'VariableNames', {'cause', 'time', 'L', 'U', 'seL', 'seU', 'IM_lo', 'IM_hi', ...
    'env_lo', 'env_hi', 'seL_iqr', 'seU_iqr', 'IMiqr_lo', 'IMiqr_hi', 'enviqr_lo', 'enviqr_hi', 'B_ok'});
writetable(TC, fullfile(dirr, sprintf('summary_cif_%s.csv', famSel)));
S = struct('freq', freq, 'freqR', freqR, 'freqP', freqP, 'eqG', eqG, 'eqR', eqR, 'eqP', eqP, ...
    'L', Lf, 'U', Uf, 'seL', seL, 'seU', seU, 'IMlo', IMlo, 'IMhi', IMhi, 'ENVlo', ENVlo, 'ENVhi', ENVhi, ...
    'seLiqr', qL, 'seUiqr', qU, 'IMiqrlo', IQlo, 'IMiqrhi', IQhi, 'ENViqrlo', EQlo, 'ENViqrhi', EQhi, ...
    'nb', nb, 'nok', nok, 'spec', spec);
end

function [IMlo, IMhi, ENVlo, ENVhi] = intervals(Lf, Uf, Ffull, seL, seU, seG, alpha)
% Imbens-Manski intervals for a point in [L, U] and the envelope interval.
[K, nt] = size(Lf);
Phi = @(x) 0.5 * erfc(-x / sqrt(2)); z = sqrt(2) * erfcinv(alpha);
IMlo = nan(K, nt); IMhi = nan(K, nt); ENVlo = nan(K, nt); ENVhi = nan(K, nt);
for k = 1:K
    for t = 1:nt
        s = max(seL(k, t), seU(k, t)); dlt = max(Uf(k, t) - Lf(k, t), 0);
        c = fzero(@(c) Phi(c + dlt / s) - Phi(-c) - (1 - alpha), [0, z + 1]);
        IMlo(k, t) = Lf(k, t) - c * seL(k, t); IMhi(k, t) = Uf(k, t) + c * seU(k, t);
        ENVlo(k, t) = min(squeeze(Ffull(k, t, :) - z * seG(k, t, :)));
        ENVhi(k, t) = max(squeeze(Ffull(k, t, :) + z * seG(k, t, :)));
    end
end
end

function s = iqrse(A, d)
% Normalized interquartile range along dimension d (linear interpolation
% between order statistics, as numpy.quantile), base MATLAB only.
A = sort(A, d); m = size(A, d);
q = @(pr) interp(A, d, 1 + pr * (m - 1));
s = (q(0.75) - q(0.25)) / 1.3489795;
end

function v = interp(A, d, h)
lo = floor(h); hi = min(lo + 1, size(A, d)); w = h - lo;
idx = repmat({':'}, 1, ndims(A));
idx{d} = lo; a = A(idx{:}); idx{d} = hi; b = A(idx{:});
v = (1 - w) * a + w * b;
end

function A = squeeze4(A)
% K x nt x 1 x B -> K x nt x B
A = reshape(A, size(A, 1), size(A, 2), []);
end
