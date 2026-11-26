function [fpr, tpr] = roc_curve(y, s)
%ROC_CURVE False/true positive rates for all thresholds (descending score).

    y = y(:); s = s(:);
    [s_sorted, order] = sort(s, 'descend');
    y = y(order);
    tp = cumsum(y == 1);
    fp = cumsum(y == 0);
    % keep only the last point of each group of tied scores
    last = [diff(s_sorted) ~= 0; true];
    tpr = [0; tp(last) / max(tp(end), 1)];
    fpr = [0; fp(last) / max(fp(end), 1)];
end
