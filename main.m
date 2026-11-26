%% MAGIC Gamma Telescope: training one MLP with five optimizers
%  GD (momentum), BFGS, L-BFGS, Levenberg-Marquardt and Artificial Bee Colony,
%  all implemented from scratch, compared with stratified 5-fold CV.
%
%  Run from the repository root:   >> main
%  Outputs go to results/ (CSV tables and PNG figures).

clear; clc; close all;

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));

cfg = default_config();
[X, y] = load_magic04(fullfile(root, 'data', 'magic04.csv'));
fprintf('Loaded %d samples, %d features  (gamma %d, hadron %d)\n', ...
        size(X, 1), size(X, 2), sum(y == 1), sum(y == 0));

R = run_experiment(X, y, cfg);

out_dir = fullfile(root, 'results');
if ~isfolder(out_dir), mkdir(out_dir); end
writetable(R.per_fold, fullfile(out_dir, 'metrics_per_fold.csv'));
writetable(R.summary,  fullfile(out_dir, 'summary.csv'));
save(fullfile(out_dir, 'run_data.mat'), 'R', 'cfg');   % for replot.m
plot_results(R, out_dir);

fprintf('\n===== Mean +- std over %d folds =====\n', cfg.k_folds);
fprintf('%-7s | %-15s | %-15s | %-15s | %s\n', 'Method', 'Test acc (%)', ...
        'Test AUC', 'Train acc (%)', 'Time/fold (s)');
S = R.summary;
for a = 1:height(S)
    fprintf('%-7s | %6.2f +- %-5.2f | %.4f +- %.4f | %6.2f +- %-5.2f | %7.1f\n', ...
        S.Algorithm(a), S.TestAcc_mean(a), S.TestAcc_std(a), ...
        S.TestAUC_mean(a), S.TestAUC_std(a), ...
        S.TrainAcc_mean(a), S.TrainAcc_std(a), S.Seconds_mean(a));
end
fprintf('\nResults written to %s\n', out_dir);
