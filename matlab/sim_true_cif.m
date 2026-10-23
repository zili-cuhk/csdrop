function [F1, F2] = sim_true_cif(Z, B, grid, f2, baseline)
%SIM_TRUE_CIF Exact true CIFs on GRID for every row of Z (mirrors R/dgp.R).
%   B is 3 x p (rows event 1, event 2, dropout); only rows 1-2 are used.
%   f2       multiplier of the event-2 baseline (default 1)
%   baseline 'pc' (piecewise constant, default) or 'weibull'
%            (Lambda_k(t) = W_k t^1.5 with W1 = 0.21/2^1.5, W2 = 0.165/2^1.5)
if nargin < 4 || isempty(f2), f2 = 1; end
if nargin < 5 || isempty(baseline), baseline = 'pc'; end
grid = grid(:)';
e1 = exp(Z * B(1, :)');
e2 = exp(Z * B(2, :)') * f2;
if strcmp(char(baseline), 'weibull')
    KAP = 1.5; W1 = 0.21 / 2^1.5; W2 = 0.165 / 2^1.5;
    r = W1 * e1 + W2 * e2;
    inc = 1 - exp(-r * grid .^ KAP);
    F1 = (W1 * e1 ./ r) .* inc;
    F2 = (W2 * e2 ./ r) .* inc;
    return
end
CUT = [0 1 2 Inf];
A1 = [0.120 0.090 0.070];
A2 = [0.090 0.075 0.060];
n = size(Z, 1);
F1 = zeros(n, numel(grid));
F2 = F1;
Sprev = ones(n, 1);
for m = 1:3
    lo = CUT(m); hi = CUT(m + 1);
    r1 = A1(m) * e1; r2 = A2(m) * e2; r = r1 + r2;
    d = min(max(grid - lo, 0), hi - lo);
    inc = 1 - exp(-r * d);
    F1 = F1 + (Sprev .* r1 ./ r) .* inc;
    F2 = F2 + (Sprev .* r2 ./ r) .* inc;
    if isfinite(hi)
        Sprev = Sprev .* exp(-r * (hi - lo));
    end
end
end
