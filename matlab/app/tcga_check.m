function tcga_check()
%TCGA_CHECK Reruns the ten TCGA-LUAD analyses (Table 6 and Table S9) with csm_select.
%   Inputs are the analysis cohorts produced by prepare_tcga_luad_crdd_application
%   (written by application/export_cohorts.py to application/data/tcga_<rule>.csv; status 1 = recorded PFS event, 2 = death without
%   earlier PFS, 3 = dropout, 0 = administrative).
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..')));
datadir = fullfile(here, '..', '..', 'application', 'data');
names = {'age10', 'male', 'stage2', 'stage34', 'nodepos', 't34'};
A = {'exclude_unknown', 'frank', -0.3; 'exclude_unknown', 'frank', 0; ...
     'exclude_unknown', 'frank', 0.2; 'exclude_unknown', 'frank', 0.5; ...
     'unknown_as_admin', 'frank', 0.2; 'unknown_as_dropout', 'frank', 0.2; ...
     'exclude_unknown', 'clayton', 0.2; 'exclude_unknown', 'clayton', 0.5; ...
     'exclude_unknown', 'gumbel', 0.2; 'exclude_unknown', 'gumbel', 0.5};
for i = 1:size(A, 1)
    if ~isfile(fullfile(datadir, ['tcga_' A{i, 1} '.csv'])), continue, end
    D = readmatrix(fullfile(datadir, ['tcga_' A{i, 1} '.csv']), 'NumHeaderLines', 1);
    c = readmatrix(fullfile(datadir, ['tcga_' A{i, 1} '_cuts.csv']));
    t0 = tic;
    res = csm_select(D(:, 1), D(:, 2), D(:, 3:end), [0 c Inf], 2, A{i, 2}, A{i, 3});
    fprintf('%2d %-18s %-7s tau=%+.1f n=%d (%.1f s)\n', i, A{i, 1}, A{i, 2}, A{i, 3}, size(D, 1), toc(t0));
    for e = 1:numel(res.sel)
        s = res.sel(e); txt = '';
        [m, j] = find(s.S);
        for k = 1:numel(m)
            txt = [txt sprintf(' %s[%d] HR=%.2f', names{j(k)}, m(k), exp(s.B(m(k), j(k))))]; %#ok<AGROW>
        end
        fprintf('    gE=%.2f gP=%.3f theta=%4d ell=%.3f |%s\n', s.gE, s.gP, s.theta, s.ell, txt);
    end
end
end
