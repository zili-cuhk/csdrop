function P = csm_prep(Y, status, Z, cuts, K)
%CSM_PREP Standardize covariates and build interval exposures and counts.
%   status: 1..K event type, K+1 dropout, 0 administrative censoring.
%   cuts:   row vector 0 = k0 < k1 < ... < kM (kM may be Inf); intervals (k_{m-1}, k_m].
Y = Y(:); status = status(:);
cuts = cuts(:)';
[n, p] = size(Z);
M = numel(cuts) - 1;
E = max(min(Y, cuts(2:end)) - cuts(1:end-1), 0);          % n x M exposures E_im
iv = sum(Y > cuts(1:M), 2);                                % interval holding Y_i
iv = min(max(iv, 1), M);
Dm = zeros(K + 1, M);
for d = 1:K + 1
    idx = status == d;
    Dm(d, :) = accumarray(iv(idx), 1, [M 1])';
end
mu = mean(Z, 1);
sd = std(Z, 0, 1);
if any(sd == 0), error('constant covariate column'); end
P = struct('n', n, 'p', p, 'K', K, 'M', M, 'Y', Y, 'st', status, ...
    'E', E, 'iv', iv, 'Dm', Dm, 'Zs', (Z - mu) ./ sd, 'mu', mu, 'sd', sd, ...
    'cuts', cuts, 'n0', sum(status == (1:K + 1), 1)');
end
