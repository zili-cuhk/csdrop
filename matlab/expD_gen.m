function dat = expD_gen(n, seed, cDrop, signal)
%EXPD_GEN Experiment D data: the Exp A strong-signal design with p = 10, Frank
%copula at tau = 0.5, and an additional dropout effect cDrop for Z3 (Z3 is an
%active event-2 covariate). Mirrors gen_data() in R/dgp.R line by line.
%   dat.Y, dat.status (1 event 1, 2 event 2, 3 dropout, 0 admin), dat.Z, dat.B
%   signal = 'strong' (default, Experiment E) or 'moderate' (Experiment E, moderate version)
if nargin < 4 || isempty(signal), signal = 'strong'; end
p = 10;
rng(seed, 'twister');
Sig = 0.3 .^ abs((1:p)' - (1:p));
Z = randn(n, p) * chol(Sig);
B = sim_coefs(signal, p);
B(3, 3) = cDrop;
A1 = [0.120 0.090 0.070]; A2 = [0.090 0.075 0.060]; AC0 = [0.055 0.075 0.100];
if strcmp(signal, 'strong')
    mC = 0.770079873784685; a = 8.39305286737204;   % frozen constants, strong / tau = 0.5
else
    mC = 0.919451899640470; a = 10.10206933249560;  % frozen constants, moderate / tau = 0.5
end
e1 = exp(Z * B(1, :)'); e2 = exp(Z * B(2, :)'); eC = exp(Z * B(3, :)');
th = csm_tau2theta('frank', 0.5);
u = rand(n, 1); w = rand(n, 1);
v = -log1p(w .* expm1(-th) ./ (w + (1 - w) .* exp(-th * u))) / th;
T = inv_cumhaz(-log(u), e1 * A1 + e2 * A2);
m = 1 + (T > 1) + (T > 2);                          % interval index 1..3
A1c = A1(:); A2c = A2(:);
p1 = e1 .* A1c(m) ./ (e1 .* A1c(m) + e2 .* A2c(m));
J = 1 + (rand(n, 1) >= p1);
C = inv_cumhaz(-log(v) / mC, eC * AC0);
Y = min(min(T, C), a);
status = zeros(n, 1);
iT = T <= C & T <= a;
status(iT) = J(iT);
status(~iT & C <= a) = 3;
dat = struct('Y', Y, 'status', status, 'Z', Z, 'B', B);
end

function t = inv_cumhaz(h, R)
c1 = R(:, 1); c2 = c1 + R(:, 2);
t = h ./ R(:, 1);
i2 = h > c1 & h <= c2;
t(i2) = 1 + (h(i2) - c1(i2)) ./ R(i2, 2);
i3 = h > c2;
t(i3) = 2 + (h(i3) - c2(i3)) ./ R(i3, 3);
end
