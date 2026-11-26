function [w_best, hist] = train_lbfgs(w0, prob, opt, es)
%TRAIN_LBFGS Limited-memory BFGS, two-loop recursion (Nocedal & Wright,
%   Alg. 7.4) with Armijo backtracking line search.

    w = w0;
    [f, g] = prob.fobj(w);
    m = opt.memory;
    S = zeros(numel(w), 0);           % columns: oldest ... newest
    Y = zeros(numel(w), 0);
    hist = init_history(opt.max_iter);

    for it = 1:opt.max_iter
        p = -lbfgs_direction(g, S, Y);
        if g.' * p >= 0
            S = S(:, []); Y = Y(:, []); p = -g;
        end
        alpha0 = 1;
        if isempty(S), alpha0 = min(1, 1 / norm(g)); end

        [w_new, f_new, g_new, alpha] = line_search_armijo(prob.fobj, w, f, g, p, alpha0);
        if alpha == 0
            S = S(:, []); Y = Y(:, []);
            [w_new, f_new, g_new, alpha] = line_search_armijo(prob.fobj, w, f, g, -g, 1);
            if alpha == 0, it = it - 1; break; end %#ok<FXSET>
        end

        s  = w_new - w;
        yk = g_new - g;
        if s.' * yk > 1e-10
            S = [S, s];  Y = [Y, yk];
            if size(S, 2) > m
                S(:, 1) = []; Y(:, 1) = [];
            end
        end

        w = w_new; f = f_new; g = g_new;
        hist.train_loss(it) = f;
        hist.val_loss(it)   = prob.val_loss(w);
        if es.update(hist.val_loss(it), w, it), break; end
    end

    [w_best, hist] = finish_history(hist, it, es, w);
end

function r = lbfgs_direction(g, S, Y)
% Two-loop recursion: returns approx. H^{-1} * g.
    k = size(S, 2);
    if k == 0, r = g; return; end
    rho = 1 ./ sum(S .* Y, 1);
    a = zeros(k, 1);
    q = g;
    for i = k:-1:1
        a(i) = rho(i) * (S(:, i).' * q);
        q = q - a(i) * Y(:, i);
    end
    gamma = (S(:, k).' * Y(:, k)) / (Y(:, k).' * Y(:, k));
    r = gamma * q;
    for i = 1:k
        b = rho(i) * (Y(:, i).' * r);
        r = r + S(:, i) * (a(i) - b);
    end
end
