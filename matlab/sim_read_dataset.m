function dat = sim_read_dataset(root, sid, rep)
%SIM_READ_DATASET Read one simulated dataset written by R/generate.R.
%   dat.Y      n x 1 observed time
%   dat.status n x 1 code: 1 event 1, 2 event 2, 3 dropout, 0 administrative
%   dat.Z      n x p covariates (raw scale, as generated)
f = fullfile(root, 'data', char(sid), sprintf('rep_%04d.csv', rep));
M = readmatrix(f, 'NumHeaderLines', 1);
dat.Y = M(:, 1);
dat.status = M(:, 2);
dat.Z = M(:, 3:end);
end
