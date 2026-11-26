function [w_best, hist] = finish_history(hist, n_iter, es, w_last)
%FINISH_HISTORY Trim the history and return the best-validation weights.

    n_iter = max(n_iter, 0);
    hist.train_loss = hist.train_loss(1:n_iter);
    hist.val_loss   = hist.val_loss(1:n_iter);
    hist.n_iter     = n_iter;
    hist.best_iter  = es.best_iter;
    if isempty(es.best_w)
        w_best = w_last;          % never improved (e.g. stopped at iteration 0)
    else
        w_best = es.best_w;
    end
end
