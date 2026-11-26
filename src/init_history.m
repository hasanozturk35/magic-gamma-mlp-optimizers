function hist = init_history(max_iter)
%INIT_HISTORY Pre-allocated loss history shared by all trainers.

    hist.train_loss = nan(max_iter, 1);
    hist.val_loss   = nan(max_iter, 1);
end
