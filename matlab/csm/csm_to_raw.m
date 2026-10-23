function [B, a] = csm_to_raw(P, X, xi)
%CSM_TO_RAW Map standardized-scale (X, xi) to the raw covariate scale.
%   z_s'x = z'(x./sd) - mu*(x./sd)', so B = X ./ sd and the baseline heights
%   absorb exp(-mu * B').
B = X ./ P.sd;
a = exp(xi) .* exp(-(B * P.mu'));
end
