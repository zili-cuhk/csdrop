function opt = csm_options(varargin)
%CSM_OPTIONS Numerical specification of Supplement S4 (manuscript v14).
%   opt = csm_options('name', value, ...) overrides individual fields.
opt = struct( ...
    'gP_grid',        [0 0.025 0.05 0.075 0.10 0.125], ... % Gamma_P in (17)
    'V',              [50 200 800], ...                    % GSELO shapes
    'eta',            0.05, ...                            % threshold (23), standardized scale
    'gE',             [0 0.25 0.5], ...                    % EBIC coefficients in (25)
    'max_iter',       600, ...                             % penalized fit
    'n_consec',       3, ...
    'tol_obj',        1e-7, ...
    'tol_coef',       1e-5, ...
    'tol_grad',       5e-3, ...
    'max_bt',         50, ...
    'mono',           1e-10, ...                           % monotonicity allowance
    'L_floor',        1e-6, ...                            % curvature floor
    'coef_bound',     4, ...                               % |x| safeguard (standardized)
    'a_lo',           1e-5, ...                            % baseline-height safeguards
    'a_hi',           2, ...
    'eps_num',        1e-8, ...                            % epsilon_num in (24)
    'refit_max_iter', 1000, ...                            % unpenalized fits
    'refit_tol',      5e-4, ...
    'min_events',     5, ...                               % per margin, overall
    'ridge_lam',      1, ...                               % weak-ridge seed: g + lam*||x||^2/n
    'tie_rel',        1e-10, ...                           % EBIC tie tolerance
    'L_relax',        1.2, ...                             % L_r starts from L_{r-1}/L_relax
    'n_starts',       2, ...                               % starts per penalized fit
    'jitter_sd',      0.03, ...                            % coefficient jitter of start 2
    'jitter_seed',    20261011);                           % base seed of the jitter
for i = 1:2:numel(varargin)
    if ~isfield(opt, varargin{i}), error('unknown option %s', varargin{i}); end
    opt.(varargin{i}) = varargin{i + 1};
end
end
