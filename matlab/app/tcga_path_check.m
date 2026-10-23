function tcga_path_check()
%TCGA_PATH_CHECK csm_path on the primary TCGA-LUAD cohort over the Frank grid
%   tau in {-0.3, 0, 0.2, 0.5}; prints robust/possible supports and CIF bounds.
%   Compare with tcga_path_expected_python.txt.
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, '..')));
datadir = fullfile(here, '..', '..', 'application', 'data');
names = {'age10', 'male', 'stage2', 'stage34', 'nodepos', 't34'};
D = readmatrix(fullfile(datadir, 'tcga_exclude_unknown.csv'), 'NumHeaderLines', 1);
c = readmatrix(fullfile(datadir, 'tcga_exclude_unknown_cuts.csv'));
tauGrid = [-0.3 0 0.2 0.5];
out = csm_path(D(:, 1), D(:, 2), D(:, 3:end), [0 c Inf], 2, 'frank', tauGrid, [], ...
    'gE', 0.25, 'times', [12 24 36]);
for g = 1:numel(tauGrid)
    fprintf('tau=%+.1f %s  F1=%s  F2=%s\n', tauGrid(g), out.status{g}, ...
        mat2str(out.Fbar(1, :, g), 4), mat2str(out.Fbar(2, :, g), 4));
end
[m, j] = find(out.SR); fprintf('S_R:\n');
for k = 1:numel(m), fprintf('   margin %d: %s\n', m(k), names{j(k)}); end
[m, j] = find(out.SP); fprintf('S_P:\n');
for k = 1:numel(m), fprintf('   margin %d: %s\n', m(k), names{j(k)}); end
fprintf('L (cause x time) = %s\nU (cause x time) = %s\n', mat2str(out.L, 4), mat2str(out.U, 4));
end
