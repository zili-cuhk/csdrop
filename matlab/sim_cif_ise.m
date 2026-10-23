function v = sim_cif_ise(Fhat, Ftrue, grid)
%SIM_CIF_ISE Integrated squared CIF error, as in the paper and R/dgp.R:
%   trapezoid over GRID divided by its length, averaged over subjects.
grid = grid(:);
sq = (Fhat - Ftrue).^2;
per = ((sq(:, 2:end) + sq(:, 1:end-1)) / 2) * diff(grid);
v = mean(per) / (max(grid) - min(grid));
end
