function [w_best, hist] = train_bfgs(w0, prob, opt, es)
%TRAIN_BFGS Quasi-Newton BFGS with a dense inverse-Hessian approximation
%   and Armijo backtracking line search (Nocedal & Wright, Alg. 6.1).

    w = w0;
    [f, g] = prob.fobj(w);
    P = numel(w);
    Hinv = eye(P);
    scaled = false;
    hist = init_history(opt.max_iter);

    for it = 1:opt.max_iter
        p = -Hinv * g;
        if g.' * p >= 0              % lost positive definiteness: restart
            Hinv = eye(P); p = -g;
        end
        alpha0 = 1;
        if ~scaled, alpha0 = min(1, 1 / norm(g)); end

        [w_new, f_new, g_new, alpha] = line_search_armijo(prob.fobj, w, f, g, p, alpha0);
        if alpha == 0                 % no progress: try steepest descent once
            Hinv = eye(P);
            [w_new, f_new, g_new, alpha] = line_search_armijo(prob.fobj, w, f, g, -g, 1);
            if alpha == 0, it = it - 1; break; end %#ok<FXSET>
        end

        s  = w_new - w;
        yk = g_new - g;
        sy = s.' * yk;
        if sy > 1e-10                 % curvature condition
            if ~scaled                % initial scaling, N&W eq. 6.20
                Hinv = (sy / (yk.' * yk)) * eye(P);
                scaled = true;
            end
            rho = 1 / sy;
            Hy  = Hinv * yk;
            Hinv = Hinv - rho * (s * Hy.' + Hy * s.') ...
                        + (rho^2 * (yk.' * Hy) + rho) * (s * s.');
        end

        w = w_new; f = f_new; g = g_new;
        hist.train_loss(it) = f;
        hist.val_loss(it)   = prob.val_loss(w);
        if es.update(hist.val_loss(it), w, it), break; end
    end

    [w_best, hist] = finish_history(hist, it, es, w);
end
