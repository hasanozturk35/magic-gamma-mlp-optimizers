%% Re-draw the figures from the last run without training again.
%  main.m saves the raw results to results/run_data.mat (not committed).

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));
data = fullfile(root, 'results', 'run_data.mat');
if ~isfile(data)
    error('replot:missing', 'Run main.m first (%s not found).', data);
end
S = load(data, 'R');
plot_results(S.R, fullfile(root, 'results'));
fprintf('Figures written to %s\n', fullfile(root, 'results'));
