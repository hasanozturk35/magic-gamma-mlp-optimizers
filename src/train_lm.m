function [w_best, hist] = train_lm(w0, prob, opt, es)
%TRAIN_LM Levenberg-Marquardt for cross-entropy loss.
%   Uses the Gauss-Newton (Fisher) matrix of the logistic output,
%       A = J' * diag(p .* (1 - p)) * J / N + lambda * diag(weight_mask),
%   where J is the N x P Jacobian of the logits, and solves
%       (A + mu*I) * delta = -grad.
%   mu is decreased after a successful step and increased after a failure.

    w  = w0;
    [f, g] = prob.fobj(w);
    P  = numel(w);
    N  = size(prob.Xtr, 1);
    R  = prob.lambda * diag(mlp_reg_mask(prob.D, prob.H));
    mu = opt.mu0;
    hist = init_history(opt.max_iter);

    for it = 1:opt.max_iter
        [J, p] = mlp_jacobian(w, prob.Xtr, prob.H);
        A = (J.' * (J .* (p .* (1 - p)))) / N + R;
        A = (A + A.') / 2;

        accepted = false;
        while mu <= opt.mu_max
            delta = -(A + mu * eye(P)) \ g;
            [f_new, g_new] = prob.fobj(w + delta);
            if isfinite(f_new) && f_new < f
                accepted = true;
                mu = max(mu * opt.mu_dec, 1e-12);
                break;
            end
            mu = mu * opt.mu_inc;
        end
        if ~accepted, it = it - 1; break; end %#ok<FXSET>

        w = w + delta; f = f_new; g = g_new;
        hist.train_loss(it) = f;
        hist.val_loss(it)   = prob.val_loss(w);
        if es.update(hist.val_loss(it), w, it), break; end
    end

    [w_best, hist] = finish_history(hist, it, es, w);
end
