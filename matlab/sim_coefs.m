function B = sim_coefs(signal, p)
%SIM_COEFS True coefficients (rows event 1, event 2, dropout); mirrors R/dgp.R.
B = zeros(3, p);
switch char(signal)
    case 'moderate'
        B(1, [1 2 5]) = [0.50 -0.40 0.30];
        B(2, [1 3 6]) = [-0.35 0.45 -0.30];
        B(3, [2 4 7]) = [0.40 -0.35 0.30];
    case 'strong'
        B(1, [1 2 5]) = [0.70 -0.70 0.60];
        B(2, [1 3 6]) = [-0.70 0.70 -0.60];
        B(3, [2 4 7]) = [0.70 -0.70 0.60];
    otherwise
        error('unknown signal %s', signal);
end
end
