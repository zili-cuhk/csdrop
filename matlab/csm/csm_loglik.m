function [ell, gX, gxi, ok, R] = csm_loglik(P, cop, X, xi, needGrad)
%CSM_LOGLIK Observed-data log likelihood (9) and its analytic scores (14)-(16).
%   X  : (K+1) x p coefficients on the standardized scale (rows: events, dropout)
%   xi : (K+1) x M log baseline heights
%   gX : d ell / d X     gxi : d ell / d xi = D - a .* R     R : R_dm of Section 3.2
if nargin < 5, needGrad = true; end
K = P.K; st = P.st;
gX = []; gxi = []; R = [];
a = exp(xi);
eta = P.Zs * X';                       % n x (K+1)
H = exp(eta) .* (P.E * a');            % n x (K+1) cumulative hazards at Y_i
lu = -sum(H(:, 1:K), 2);
lv = -H(:, K + 1);
u = exp(lu); v = exp(lv);
if any(u <= 0) || any(v <= 0)
    ell = -Inf; ok = false; return
end
q = csm_copula(cop, u, v);
isE = st >= 1 & st <= K;
isC = st == K + 1;
isA = st == 0;
ell_i = zeros(P.n, 1);
for d = 1:K + 1
    idx = st == d;
    ell_i(idx) = ell_i(idx) + xi(sub2ind(size(xi), d * ones(nnz(idx), 1), P.iv(idx))) + eta(idx, d);
end
ell_i(isE) = ell_i(isE) + lu(isE) + q.logC1(isE);
ell_i(isC) = ell_i(isC) + lv(isC) + q.logC2(isC);
ell_i(isA) = ell_i(isA) + q.logC(isA);
ell = sum(ell_i);
if ~isfinite(ell)
    ell = -Inf; ok = false; return
end
ok = true;
if ~needGrad, return, end
rT = q.tA; rT(isE) = q.tE(isE); rT(isC) = q.tD(isC);
rC = q.cA; rC(isE) = q.cE(isE); rC(isC) = q.cD(isC);
r = [repmat(rT, 1, K), rC];                        % n x (K+1)
Delta = double(st == (1:K + 1));                   % n x (K+1)
gX = (Delta - r .* H)' * P.Zs;                     % (K+1) x p
R = (r .* exp(eta))' * P.E;                        % (K+1) x M
gxi = P.Dm - a .* R;
ok = all(isfinite(gX(:))) && all(isfinite(gxi(:)));
end
