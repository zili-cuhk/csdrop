function cuts = sim_cuts(spec, baseline, Y, status)
%SIM_CUTS Baseline partition for one fit.
%   spec 'auto' : (0, 1, 2, Inf) for the piecewise-constant designs, and the
%                 tertiles of the pooled event and dropout times for the
%                 Weibull design (three intervals, as in the primary design)
%   spec 'q3', 'q5' : tertiles or quintiles of the pooled event and dropout
%                 times, giving 3 or 5 intervals
spec = char(spec);
if isempty(spec) || strcmp(spec, 'auto')
    if strcmp(char(baseline), 'weibull'), spec = 'q3'; else, cuts = [0 1 2 Inf]; return, end
end
M = str2double(spec(2:end));
if ~startsWith(spec, 'q') || isnan(M) || M < 2, error('unknown cuts spec %s', spec); end
t = sort(Y(status >= 1 & status <= 3));
nt = numel(t);                                  % empirical quantiles without the
q = t(max(1, ceil((1:M - 1) / M * nt)));        % Statistics Toolbox (type 1)
cuts = [0 q(:)' Inf];
end
