%% Re-draw the figures from the last run without training again.
%  main.m saves the raw results to results/run_data.mat (not committed).
%
%  Only load a run_data.mat that YOUR OWN main.m produced: like any .mat
%  file, it could contain objects whose class code runs on load.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));
data = fullfile(root, 'results', 'run_data.mat');
if ~isfile(data)
    error('replot:missing', 'Run main.m first (%s not found).', data);
end

% inspect the file before loading anything from it
info = whos('-file', data);
if ~any(strcmp({info.name}, 'R')) || ~strcmp(info(strcmp({info.name}, 'R')).class, 'struct')
    error('replot:format', 'Unexpected content in %s.', data);
end
S = load(data, 'R');
R = S.R;
needed = {'per_fold', 'summary', 'algorithms', 'hist', 'scores', 'labels'};
if ~all(isfield(R, needed)) || ~istable(R.per_fold) || ~istable(R.summary)
    error('replot:format', 'Unexpected content in %s.', data);
end

plot_results(R, fullfile(root, 'results'));
fprintf('Figures written to %s\n', fullfile(root, 'results'));
