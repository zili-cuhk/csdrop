function F = csm_cif(P, X, xi, Zraw, grid)
%CSM_CIF Exact intervalwise cause-specific CIFs (Section 3.5).
%   X, xi : standardized-scale fit; Zraw : n x p raw covariates; grid : times.
%   F     : 1 x K cell, F{k} is n x numel(grid).
K = P.K; cuts = P.cuts;
grid = grid(:)';
Zs = (Zraw - P.mu) ./ P.sd;
a = exp(xi);
lam = exp(Zs * X(1:K, :)');                 % n x K
n = size(Zs, 1);
F = repmat({zeros(n, numel(grid))}, 1, K);
Sprev = ones(n, 1);
for m = 1:P.M
    lo = cuts(m); hi = cuts(m + 1);
    rk = lam .* a(1:K, m)';                 % n x K interval hazards
    r = sum(rk, 2);
    d = max(min(grid, hi) - lo, 0);
    inc = -expm1(-r * d) ./ r;               % (1 - exp(-r d)) / r
    z = r == 0;
    if any(z), inc(z, :) = repmat(d, nnz(z), 1); end
    for k = 1:K
        F{k} = F{k} + (Sprev .* rk(:, k)) .* inc;
    end
    if isfinite(hi)
        Sprev = Sprev .* exp(-r * (hi - lo));
    end
end
end
