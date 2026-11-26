function [w_best, hist] = train_abc(w0, prob, opt, es)
%TRAIN_ABC Artificial Bee Colony (Karaboga, 2005) for MLP weights.
%   Gradient-free: only loss values are used. Additions to the classic
%   algorithm (Akay & Karaboga, 2012):
%     * modification rate: each move perturbs ~mod_rate of the weights
%       instead of a single one, which matters in hundreds of dimensions;
%     * sources are initialised (and scouts re-initialised) with the same
%       Xavier initialiser as the other methods, and one source is w0.
%   Fitness for the onlooker roulette is the classic 1 / (1 + loss).

    P   = numel(w0);
    SN  = opt.food_number;
    B   = opt.bound;
    n_change = max(1, round(opt.mod_rate * P));

    foods = zeros(SN, P);
    foods(1, :) = w0.';
    for i = 2:SN
        foods(i, :) = mlp_init(prob.D, prob.H).';
    end
    foods = min(max(foods, -B), B);

    cost = zeros(SN, 1);
    for i = 1:SN
        cost(i) = prob.fobj(foods(i, :).');
    end
    trial = zeros(SN, 1);
    [best_cost, ib] = min(cost);
    best = foods(ib, :);

    hist = init_history(opt.max_cycle);

    for it = 1:opt.max_cycle
        % --- employed bees: one move per source
        for i = 1:SN
            try_move(i);
        end

        % --- onlooker bees: sources chosen proportional to fitness
        fit  = 1 ./ (1 + cost);
        cdf  = cumsum(fit) / sum(fit);
        for n = 1:SN
            i = find(rand <= cdf, 1, 'first');
            try_move(i);
        end

        % --- scout bee: abandon the most-tried exhausted source
        [tmax, i] = max(trial);
        if tmax > opt.limit
            foods(i, :) = min(max(mlp_init(prob.D, prob.H).', -B), B);
            cost(i) = prob.fobj(foods(i, :).');
            trial(i) = 0;
        end

        [c, ib] = min(cost);
        if c < best_cost
            best_cost = c;
            best = foods(ib, :);
        end

        hist.train_loss(it) = best_cost;
        hist.val_loss(it)   = prob.val_loss(best.');
        if isfield(opt, 'tag') && mod(it, 250) == 0
            fprintf('    %s: cycle %4d / %d, val loss %.4f\n', ...
                    opt.tag, it, opt.max_cycle, hist.val_loss(it));
        end
        if es.update(hist.val_loss(it), best.', it), break; end
    end

    [w_best, hist] = finish_history(hist, it, es, best.');

    function try_move(i)
        k = randi(SN - 1);
        if k >= i, k = k + 1; end              % partner k ~= i
        j = randperm(P, n_change);
        v = foods(i, :);
        phi = 2 * rand(1, n_change) - 1;
        v(j) = v(j) + phi .* (foods(i, j) - foods(k, j));
        v = min(max(v, -B), B);
        cv = prob.fobj(v.');
        if cv < cost(i)                        % greedy selection
            foods(i, :) = v;
            cost(i) = cv;
            trial(i) = 0;
        else
            trial(i) = trial(i) + 1;
        end
    end
end
