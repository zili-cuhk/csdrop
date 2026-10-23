function ok = csm_selftest()
%CSM_SELFTEST Verification checks of Section 3.6 plus agreement with the
%reference prototype on a fixed dataset (test/ref_data.csv, ref_values.json).
%   Run from any folder:  addpath(genpath('matlab')); csm_selftest
here = fileparts(mfilename('fullpath'));
tdir = fullfile(here, 'test');
ok = true;

% ---- 1. copula derivatives vs central finite differences
rng(0);
u = 0.05 + 0.93 * rand(200, 1); v = 0.05 + 0.93 * rand(200, 1);
fams = {'frank', 0.2; 'frank', 0.5; 'frank', -0.3; 'clayton', 0.5; 'gumbel', 0.5; 'indep', 0};
for i = 1:size(fams, 1)
    cop = struct('family', fams{i, 1}, 'theta', csm_tau2theta(fams{i, 1}, fams{i, 2}));
    q = csm_copula(cop, u, v);
    e = 1e-6;
    L = @(key, uu, vv) getfield(csm_copula(cop, uu, vv), key); %#ok<GFLD>
    up = u * exp(e); um = u * exp(-e); vp = v * exp(e); vm = v * exp(-e);
    fd.tE = 1 + (L('logC1', up, v) - L('logC1', um, v)) / (2 * e);
    fd.tD = (L('logC2', up, v) - L('logC2', um, v)) / (2 * e);
    fd.tA = (L('logC', up, v) - L('logC', um, v)) / (2 * e);
    fd.cE = (L('logC1', u, vp) - L('logC1', u, vm)) / (2 * e);
    fd.cD = 1 + (L('logC2', u, vp) - L('logC2', u, vm)) / (2 * e);
    fd.cA = (L('logC', u, vp) - L('logC', u, vm)) / (2 * e);
    h = 1e-6;
    C1fd = (exp(L('logC', u + h, v)) - exp(L('logC', u - h, v))) / (2 * h);
    C2fd = (exp(L('logC', u, v + h)) - exp(L('logC', u, v - h))) / (2 * h);
    err = max([max(abs(exp(q.logC1) - C1fd)), max(abs(exp(q.logC2) - C2fd)), ...
        cellfun(@(k) max(abs(q.(k) - fd.(k))), fieldnames(fd))']);
    ok = report(sprintf('copula %s tau=%+.1f', fams{i, 1}, fams{i, 2}), err < 1e-6, err) && ok;
end

% ---- 2. reference data and likelihood values
M = readmatrix(fullfile(tdir, 'ref_data.csv'), 'NumHeaderLines', 1);
Y = M(:, 1); st = M(:, 2); Z = M(:, 3:end);
ref = jsondecode(fileread(fullfile(tdir, 'ref_values.json')));
cuts = [0 1 2 Inf];
P = csm_prep(Y, st, Z, cuts, 2);
X = 0.3 * sin((1:3)' * (1:10));
xi = log([0.12 0.09 0.07; 0.09 0.075 0.06; 0.05 0.07 0.09]);
for i = 1:numel(ref.lik)
    r = ref.lik(i);
    cop = struct('family', r.family, 'theta', csm_tau2theta(r.family, r.tau));
    [ell, gX, gxi] = csm_loglik(P, cop, X, xi, true);
    dl = abs(ell - r.ell) / abs(r.ell);
    dg = max([max(abs(gX - r.gX), [], 'all'), max(abs(gxi - r.gxi), [], 'all')]);
    ok = report(sprintf('loglik vs reference %s tau=%+.1f', r.family, r.tau), ...
        dl < 1e-10 && dg < 1e-7 && abs(cop.theta - r.theta) < 1e-9, max(dl, dg)) && ok;
    % analytic scores vs finite differences (eqs 14-16)
    hh = 1e-6; e2 = 0;
    for dj = [1 1; 2 4; 3 7]'
        Xp = X; Xp(dj(1), dj(2)) = Xp(dj(1), dj(2)) + hh;
        Xm = X; Xm(dj(1), dj(2)) = Xm(dj(1), dj(2)) - hh;
        fdv = (csm_loglik(P, cop, Xp, xi, false) - csm_loglik(P, cop, Xm, xi, false)) / (2 * hh);
        e2 = max(e2, abs(fdv - gX(dj(1), dj(2))) / max(1, abs(fdv)));
    end
    for dm = [1 1; 2 3; 3 2]'
        xp = xi; xp(dm(1), dm(2)) = xp(dm(1), dm(2)) + hh;
        xm = xi; xm(dm(1), dm(2)) = xm(dm(1), dm(2)) - hh;
        fdv = (csm_loglik(P, cop, X, xp, false) - csm_loglik(P, cop, X, xm, false)) / (2 * hh);
        e2 = max(e2, abs(fdv - gxi(dm(1), dm(2))) / max(1, abs(fdv)));
    end
    ok = report(sprintf('scores vs finite differences %s', r.family), e2 < 1e-6, e2) && ok;
end

% ---- 3. independence factorization and relabelling invariance
cop0 = struct('family', 'indep', 'theta', 0);
ell0 = csm_loglik(P, cop0, X, xi, false);
eta = P.Zs * X'; H = exp(eta) .* (P.E * exp(xi)');
tot = 0;
for d = 1:3
    idx = st == d;
    tot = tot + sum(xi(sub2ind(size(xi), d * ones(nnz(idx), 1), P.iv(idx))) + eta(idx, d)) - sum(H(:, d));
end
ok = report('independence factorization', abs(ell0 - tot) < 1e-8, abs(ell0 - tot)) && ok;
st2 = st; st2(st == 1) = 2; st2(st == 2) = 1;
P2 = csm_prep(Y, st2, Z, cuts, 2);
copF = struct('family', 'frank', 'theta', csm_tau2theta('frank', 0.5));
dr = csm_loglik(P, copF, X, xi, false) - csm_loglik(P2, copF, X([2 1 3], :), xi([2 1 3], :), false);
ok = report('event relabelling invariance', abs(dr) < 1e-8, abs(dr)) && ok;

% ---- 4. full Algorithm 1 vs reference (Frank tau = 0.5)
t0 = tic;
res = csm_select(Y, st, Z, cuts, 2, 'frank', 0.5);
secs = toc(t0);
ok = report(sprintf('Algorithm 1 runs (%.1f s)', secs), strcmp(res.status, 'ok'), 0) && ok;
if strcmp(res.status, 'ok')
    for e = 1:numel(ref.select)
        r = ref.select(e); s = res.sel(e);
        same = isequal(double(s.S), r.S) && s.gP == r.gP && s.theta == r.theta;
        dB = max(abs(s.B - r.B), [], 'all');
        ok = report(sprintf('selection gE=%.2f: support, (gP, theta), coefficients', r.gE), ...
            same && abs(s.ell - r.ell) < 1e-4 && dB < 1e-3, dB) && ok;
    end
    F = csm_cif(res.P, res.sel(2).X, res.sel(2).xi, Z(1:5, :), [0.5 1 3 6]);
    Fr = ref.cif_first5;
    dF = max([max(abs(F{1} - squeeze(Fr(1, :, :))), [], 'all'), ...
        max(abs(F{2} - squeeze(Fr(2, :, :))), [], 'all')]);
    ok = report('CIF vs reference', dF < 1e-4, dF) && ok;
end
if ok, fprintf('\nALL SELF-TESTS PASSED\n'); else, fprintf('\nSOME SELF-TESTS FAILED\n'); end
end

function ok = report(label, pass, val)
if pass, tag = 'PASS'; else, tag = 'FAIL'; end
fprintf('[%s] %-55s %.2e\n', tag, label, val);
ok = pass;
end
