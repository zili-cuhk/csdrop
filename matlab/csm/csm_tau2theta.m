function th = csm_tau2theta(family, tau)
%CSM_TAU2THETA Kendall's tau -> copula parameter (Supplement S1).
switch lower(char(family))
    case 'indep'
        th = 0;
    case 'frank'
        if tau == 0, th = 0; return; end
        f = @(t) 1 - 4 ./ t + 4 * debye1(t) ./ t - tau;
        if tau > 0, br = [1e-6 200]; else, br = [-200 -1e-6]; end
        th = fzero(f, br, optimset('TolX', 1e-14));
    case 'clayton'
        th = 2 * tau / (1 - tau);
    case 'gumbel'
        th = 1 / (1 - tau);
    otherwise
        error('unknown copula family %s', family);
end
end

function d = debye1(t)
g = @(s) s ./ expm1(s);
d = integral(@(s) fill0(g, s), 0, t, 'RelTol', 1e-13, 'AbsTol', 1e-15) / t;
end

function y = fill0(g, s)
y = g(s);
y(s == 0) = 1;
end
