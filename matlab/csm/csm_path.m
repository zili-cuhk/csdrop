function out = csm_path(Y, status, Z, cuts, K, family, tauGrid, opt, varargin)
%CSM_PATH Algorithm 1 over a grid of association values (sensitivity set).
%   Runs csm_select at every Kendall's tau in tauGrid and returns the
%   thresholded supports, their intersection (robust support) and union
%   (possible support), and grid bounds on the cohort-averaged CIFs.
%
%   out = csm_path(Y, status, Z, cuts, K, family, tauGrid)
%   out = csm_path(..., opt, 'gE', 0.25, 'times', [12 24 36])
%
%   tauGrid must include both endpoints of the sensitivity set.
%   out.tau      1 x G grid
%   out.status   1 x G cell of csm_select status strings
%   out.S        (K+1) x p x G logical thresholded supports (NaN-free; a
%                grid point whose fit failed is marked in out.ok)
%   out.ok       1 x G logical, fit available
%   out.SR, out.SP   (K+1) x p logical robust and possible supports over
%                the grid points with out.ok
%   out.Fbar     K x numel(times) x G cohort-averaged CIFs of the refits
%   out.L, out.U K x numel(times) grid bounds; out.gL, out.gU argmin/argmax
%   out.fits     1 x G cell with the selected refit (X, xi, B, a, gP, theta)
if nargin < 8 || isempty(opt), opt = csm_options(); end
ip = inputParser;
ip.addParameter('gE', 0.25);
ip.addParameter('times', []);
ip.parse(varargin{:});
gE = ip.Results.gE; times = ip.Results.times(:)';
if isempty(times)
    ye = sort(Y(status >= 1 & status <= K));
    times = ye(max(1, round([0.25 0.5 0.75] * numel(ye))))';
end

G = numel(tauGrid);
p = size(Z, 2); D1 = K + 1;
out = struct('tau', tauGrid(:)', 'gE', gE, 'times', times, 'status', {cell(1, G)}, ...
    'S', false(D1, p, G), 'ok', false(1, G), 'SR', [], 'SP', [], ...
    'Fbar', nan(K, numel(times), G), 'L', [], 'U', [], 'gL', [], 'gU', [], 'fits', {cell(1, G)});
for g = 1:G
    res = csm_select(Y, status, Z, cuts, K, family, tauGrid(g), opt);
    out.status{g} = res.status;
    if ~strcmp(res.status, 'ok'), continue; end
    j = find(abs([res.sel.gE] - gE) < 1e-12, 1);
    if isempty(j), error('gE = %g is not in opt.gE', gE); end
    s = res.sel(j);
    out.ok(g) = true;
    out.S(:, :, g) = s.S;
    F = csm_cif(res.P, s.X, s.xi, Z, times);
    for k = 1:K
        out.Fbar(k, :, g) = mean(F{k}, 1);
    end
    out.fits{g} = struct('X', s.X, 'xi', s.xi, 'B', s.B, 'a', s.a, 'gP', s.gP, 'theta', s.theta);
end
okg = find(out.ok);
if isempty(okg), return; end
out.SR = all(out.S(:, :, okg), 3);
out.SP = any(out.S(:, :, okg), 3);
[out.L, iL] = min(out.Fbar(:, :, okg), [], 3);
[out.U, iU] = max(out.Fbar(:, :, okg), [], 3);
out.gL = okg(iL); out.gU = okg(iU);
end
