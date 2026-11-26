function is_val = stratified_holdout(y, ratio)
%STRATIFIED_HOLDOUT Logical mask selecting ~ratio of each class for validation.

    y = y(:);
    is_val = false(numel(y), 1);
    for c = unique(y)'
        idx = find(y == c);
        n_val = round(ratio * numel(idx));
        idx = idx(randperm(numel(idx), n_val));
        is_val(idx) = true;
    end
end
