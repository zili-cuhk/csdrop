function bb = csm_boot_bounds(Y, status, Z, cuts, K, family, tauGrid, opt, B, seed, alpha, varargin)
%CSM_BOOT_BOUNDS Bootstrap for csm_path: selection frequencies over the
%   grid, frequencies of membership in the robust and possible supports, and
%   Imbens-Manski intervals for the cohort-averaged CIF bounds.
%
%   bb = csm_boot_bounds(Y, status, Z, cuts, K, family, tauGrid, opt, B, seed, alpha)
%   Extra name-value pairs are passed to csm_path ('gE', 'times').
%
%   The partition (cuts) and tuning grids are held fixed across resamples.
%   The IM interval (Imbens and Manski, 2004; Stoye, 2009) uses bootstrap
%   standard errors of the lower and upper bounds. It is valid when the
%   argmin and argmax over the grid are unique (e.g. a monotone CIF); out.tieL
%   and out.tieU flag resamples in which the argmin/argmax moved.
if nargin < 8 || isempty(opt), opt = csm_options(); end
if nargin < 9 || isempty(B), B = 200; end
if nargin < 10 || isempty(seed), seed = 20261012; end
if nargin < 11 || isempty(alpha), alpha = 0.05; end
full = csm_path(Y, status, Z, cuts, K, family, tauGrid, opt, varargin{:});
n = numel(Y); G = numel(tauGrid);
[D1, p] = size(full.S(:, :, 1));
nt = numel(full.times);
cnt = zeros(D1, p, G); nOK = zeros(1, G);
inR = zeros(D1, p); inP = zeros(D1, p); nAll = 0;
Lb = nan(K, nt, B); Ub = nan(K, nt, B);
tieL = 0; tieU = 0;
rng(seed, 'twister');
idxAll = ceil(n * rand(n, B));          % same convention as the TCGA bootstrap
for b = 1:B
    idx = idxAll(:, b);
    pb = csm_path(Y(idx), status(idx), Z(idx, :), cuts, K, family, tauGrid, opt, ...
        'gE', full.gE, 'times', full.times);
    for g = 1:G
        if pb.ok(g), cnt(:, :, g) = cnt(:, :, g) + pb.S(:, :, g); nOK(g) = nOK(g) + 1; end
    end
    if all(pb.ok)
        nAll = nAll + 1;
        inR = inR + pb.SR; inP = inP + pb.SP;
        Lb(:, :, b) = pb.L; Ub(:, :, b) = pb.U;
        tieL = tieL + any(pb.gL(:) ~= full.gL(:));
        tieU = tieU + any(pb.gU(:) ~= full.gU(:));
    end
end
bb = struct();
bb.full = full;
bb.freqS = cnt ./ reshape(max(nOK, 1), 1, 1, G);   % selection frequency per grid point
bb.nOK = nOK;
bb.freqSR = inR / max(nAll, 1);
bb.freqSP = inP / max(nAll, 1);
bb.nAll = nAll;
sL = std(Lb, 0, 3, 'omitnan'); sU = std(Ub, 0, 3, 'omitnan');
bb.seL = sL; bb.seU = sU;
bb.ciLo = nan(K, nt); bb.ciHi = nan(K, nt);
for k = 1:K
    for t = 1:nt
        s = max(sL(k, t), sU(k, t));
        delta = max(full.U(k, t) - full.L(k, t), 0);
        if ~(s > 0), continue; end
        Phi = @(x) 0.5 * erfc(-x / sqrt(2));
        f = @(c) Phi(c + delta / s) - Phi(-c) - (1 - alpha);
        c = fzero(f, [0, sqrt(2) * erfcinv(alpha) + 1]);
        bb.ciLo(k, t) = full.L(k, t) - c * sL(k, t);
        bb.ciHi(k, t) = full.U(k, t) + c * sU(k, t);
    end
end
bb.tieL = tieL / max(nAll, 1); bb.tieU = tieU / max(nAll, 1);
bb.alpha = alpha; bb.B = B; bb.seed = seed;
end
