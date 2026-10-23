function q = csm_copula(cop, u, v)
%CSM_COPULA Survival-copula quantities at (u, v) for the likelihood (9) and
%the log-derivative ratios (12)-(13).
%   cop.family : 'indep' | 'frank' | 'clayton' | 'gumbel';  cop.theta
%   Returns struct with column vectors
%     logC, logC1, logC2
%     tE = 1 + u C11/C1,  tD = u C12/C2,      tA = u C1/C     (r_T by record type)
%     cE = v C12/C1,      cD = 1 + v C22/C2,  cA = v C2/C     (r_C by record type)
%   All formulas were checked against central finite differences.
fam = lower(char(cop.family));
th = cop.theta;
lu = log(u); lv = log(v);
isIndep = strcmp(fam, 'indep') || ...
    (strcmp(fam, 'frank') && abs(th) < 1e-10) || ...
    (strcmp(fam, 'clayton') && abs(th) < 1e-10) || ...
    (strcmp(fam, 'gumbel') && abs(th - 1) < 1e-10);
if isIndep
    one = ones(size(u));
    q = struct('logC', lu + lv, 'logC1', lv, 'logC2', lu, ...
        'tE', one, 'tD', one, 'tA', one, 'cE', one, 'cD', one, 'cA', one);
    return
end
switch fam
    case 'frank'
        eu = exp(-th * u); ev = exp(-th * v);
        a = expm1(-th * u); b = expm1(-th * v); c = expm1(-th);
        D = c + a .* b;
        C = -log1p(a .* b / c) / th;
        q.logC  = log(C);
        q.logC1 = -th * u + log(b ./ D);
        q.logC2 = -th * v + log(a ./ D);
        q.tE = 1 - th * u .* (c - b) ./ D;
        q.tD = -th * u .* eu * c ./ (a .* D);
        q.tA = u .* eu .* b ./ (D .* C);
        q.cE = -th * v .* ev * c ./ (b .* D);
        q.cD = 1 - th * v .* (c - a) ./ D;
        q.cA = v .* ev .* a ./ (D .* C);
    case 'clayton'
        U = exp(-th * lu); V = exp(-th * lv);       % u^-th, v^-th
        S = U + V - 1;
        lS = log(S);
        q.logC  = -lS / th;
        q.logC1 = -(th + 1) * lu - (1 / th + 1) * lS;
        q.logC2 = -(th + 1) * lv - (1 / th + 1) * lS;
        q.tE = -th + (th + 1) * U ./ S;
        q.tD = (th + 1) * U ./ S;
        q.tA = U ./ S;
        q.cE = (th + 1) * V ./ S;
        q.cD = -th + (th + 1) * V ./ S;
        q.cA = V ./ S;
    case 'gumbel'
        x = -lu; y = -lv;
        A = (x .^ th + y .^ th) .^ (1 / th);
        lA = log(A);
        Ax = exp((1 - th) * lA + (th - 1) * log(x));
        Ay = exp((1 - th) * lA + (th - 1) * log(y));
        f = 1 + (th - 1) ./ A;
        q.logC  = -A;
        q.logC1 = -A + (1 - th) * lA + (th - 1) * log(x) + x;
        q.logC2 = -A + (1 - th) * lA + (th - 1) * log(y) + y;
        q.tE = Ax .* f - (th - 1) ./ x;
        q.tD = Ax .* f;
        q.tA = Ax;
        q.cE = Ay .* f;
        q.cD = Ay .* f - (th - 1) ./ y;
        q.cA = Ay;
    otherwise
        error('unknown copula family %s', fam);
end
end
