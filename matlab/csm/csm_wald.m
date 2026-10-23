function [seB, ok, Vs] = csm_wald(P, cop, X, xi, S, h)
%CSM_WALD Model-based standard errors of a support-restricted refit.
%   The observed information of (X(S), xi) is computed by central differences
%   of the analytic scores of csm_loglik, symmetrized and inverted. The
%   baseline log heights are kept as nuisance parameters, so the standard
%   errors account for their estimation.
%
%   [seB, ok] = csm_wald(P, cop, X, xi, S)
%
%   P, cop  : prepared data and copula, as in csm_fit
%   X, xi   : refit on the standardized scale ((K+1) x p and (K+1) x M)
%   S       : (K+1) x p logical support of the refit
%   seB     : (K+1) x p standard errors on the raw covariate scale
%             (NaN off the support). Because B = X ./ sd, se(B) = se(X) ./ sd.
%   ok      : false if the information matrix is not finite or not
%             positive definite
%   Vs      : covariance matrix of (X(S), xi) on the standardized scale
if nargin < 6 || isempty(h), h = 1e-5; end
[K1, M] = size(xi);
iS = find(S(:));
q = numel(iS);
d = q + K1 * M;
th0 = [X(iS); xi(:)];
Hm = zeros(d);
for j = 1:d
    e = zeros(d, 1);
    e(j) = h * max(1, abs(th0(j)));
    Hm(:, j) = (score(th0 + e) - score(th0 - e)) / (2 * e(j));
end
Hm = (Hm + Hm') / 2;
seB = nan(size(X));
Vs = [];
ok = all(isfinite(Hm(:)));
if ok
    [~, flag] = chol(-Hm);
    ok = flag == 0;
end
if ~ok, return, end
Vs = inv(-Hm);
seX = nan(size(X));
seX(iS) = sqrt(diag(Vs(1:q, 1:q)));
seB = seX ./ P.sd;

    function g = score(th)
        Xt = zeros(size(X));
        Xt(iS) = th(1:q);
        [~, gX, gxi, okl] = csm_loglik(P, cop, Xt, reshape(th(q + 1:end), K1, M), true);
        if ~okl
            g = nan(d, 1);
        else
            g = [gX(iS); gxi(:)];
        end
    end
end
