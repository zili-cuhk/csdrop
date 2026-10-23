function [val, w] = csm_penalty(P, X, pen)
%CSM_PENALTY Penalty term of (19) and its majorization weights (21).
%   pen.kind = 'none' | 'gselo' (pen.rho: (K+1) x 1, pen.theta) | 'ridge' (pen.lam)
%   val : penalty added to g = -2 ell / n
%   w   : coefficient of x^2 in the quadratic majorizer, so that the penalty
%         gradient at X is 2 * w .* X
switch pen.kind
    case 'gselo'
        e = exp(-pen.theta * X .^ 2);
        val = sum(pen.rho(:) .* (1 - e), 'all') / P.n;
        w = pen.rho(:) .* pen.theta .* e / P.n;
    case 'ridge'
        val = pen.lam * sum(X .^ 2, 'all') / P.n;
        w = pen.lam / P.n * ones(size(X));
    otherwise
        val = 0;
        w = zeros(size(X));
end
end
