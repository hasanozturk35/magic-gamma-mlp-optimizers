function fold = stratified_kfold(y, K)
%STRATIFIED_KFOLD Assign each sample to one of K folds, keeping class ratios.
%   Uses the current global RNG state (call rng() before for reproducibility).

    y = y(:);
    fold = zeros(numel(y), 1);
    classes = unique(y);
    offset = 0;
    for c = classes'
        idx = find(y == c);
        idx = idx(randperm(numel(idx)));
        % continue the round-robin across classes so fold sizes stay balanced
        fold(idx) = mod(offset + (0:numel(idx)-1), K) + 1;
        offset = offset + numel(idx);
    end
end
