function m = binary_metrics(y, p)
%BINARY_METRICS Accuracy, AUC, F1, Brier score and log-loss.
%   y: true labels {0,1}, p: predicted P(y = 1). Threshold 0.5.

    y = y(:); p = p(:);
    pred = p >= 0.5;
    tp = sum(pred & y == 1);
    fp = sum(pred & y == 0);
    fn = sum(~pred & y == 1);

    m.accuracy = mean(pred == y) * 100;
    m.auc      = auc_rank(y, p);
    m.f1       = 2 * tp / max(2 * tp + fp + fn, 1);
    m.brier    = mean((p - y).^2);
    pc = min(max(p, 1e-12), 1 - 1e-12);
    m.logloss  = -mean(y .* log(pc) + (1 - y) .* log(1 - pc));
end

function a = auc_rank(y, s)
% ROC AUC via the Mann-Whitney U statistic, with average ranks for ties.
    n1 = sum(y == 1);
    n0 = sum(y == 0);
    if n1 == 0 || n0 == 0, a = NaN; return; end
    [ss, order] = sort(s);
    r = zeros(size(s));
    r(order) = 1:numel(s);
    % average ranks within tied groups
    [~, ~, grp] = unique(ss);
    avg = accumarray(grp, (1:numel(s)).', [], @mean);
    r(order) = avg(grp);
    a = (sum(r(y == 1)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end
