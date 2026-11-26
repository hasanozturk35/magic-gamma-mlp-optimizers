function prob = make_problem(Xtr, ytr, Xva, yva, H, lambda)
%MAKE_PROBLEM Bundle a training problem and its objective functions.
%   prob.fobj(w)    -> [train loss (with L2), gradient]
%   prob.val_loss(w)-> validation cross-entropy (no L2), used for early stopping

    prob.Xtr = Xtr;  prob.ytr = ytr;
    prob.Xva = Xva;  prob.yva = yva;
    prob.H = H;
    prob.D = size(Xtr, 2);
    prob.lambda = lambda;
    prob.fobj     = @(w) mlp_loss_grad(w, Xtr, ytr, H, lambda);
    prob.val_loss = @(w) mlp_loss_grad(w, Xva, yva, H, 0);
end
