function xi = csm_breslow_init(P, X)
%CSM_BRESLOW_INIT Piecewise-exponential Breslow ratios D_dm / sum_i exp(z_i'x_d) E_im
%(log scale), used to initialize the baseline heights for a coefficient seed.
den = exp(P.Zs * X')' * P.E;               % (K+1) x M
xi = log(max(P.Dm, 0.5) ./ den);
end
