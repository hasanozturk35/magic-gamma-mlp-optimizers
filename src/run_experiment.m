function R = run_experiment(X, y, cfg)
%RUN_EXPERIMENT Stratified K-fold CV over all training algorithms.
%
%   For every outer fold:
%     1. the test fold is held out and never touched during training;
%     2. the remaining data is split (stratified) into train / validation;
%     3. the standardizer is fitted on the TRAIN part only;
%     4. every algorithm starts from the SAME initial weights w0 and stops
%        on validation loss; its best-validation weights are evaluated on
%        the test fold.
%
%   All random draws are seeded per fold and per algorithm, so the folds
%   are independent and the run is reproducible.

    rng(cfg.seed);
    K = cfg.k_folds;
    algos = cfg.algorithms;
    A = numel(algos);

    % ---- all splits and initial weights are drawn up front (serially)
    fold = stratified_kfold(y, K);
    splits = cell(K, 1);
    for k = 1:K
        is_test = fold == k;
        yrest = y(~is_test);
        s.is_test = is_test;
        s.is_val  = stratified_holdout(yrest, cfg.val_ratio);
        s.w0      = mlp_init(size(X, 2), cfg.hidden);
        splits{k} = s;
    end

    out = cell(K, 1);
    for k = 1:K
        out{k} = run_fold(X, y, splits{k}, k, cfg);
    end

    % ---- collect
    rows = {};
    R.hist   = cell(A, K);
    R.scores = cell(A, 1);
    R.labels = cell(A, 1);
    for k = 1:K
        o = out{k};
        for a = 1:A
            R.hist{a, k} = o.hist{a};
            R.scores{a} = [R.scores{a}; o.p_test{a}];
            R.labels{a} = [R.labels{a}; o.y_test];
            rows = [rows; o.rows(a, :)]; %#ok<AGROW>
        end
    end

    R.per_fold = cell2table(rows, 'VariableNames', ...
        {'Algorithm', 'Fold', 'TrainAcc', 'TestAcc', 'TestAUC', 'TestF1', ...
         'TestBrier', 'TestLogLoss', 'Iterations', 'BestIter', 'Seconds'});
    R.summary = summarize(R.per_fold, algos);
    R.algorithms = algos;
end

function o = run_fold(X, y, s, k, cfg)
    algos = cfg.algorithms;
    A = numel(algos);

    Xrest = X(~s.is_test, :);  yrest = y(~s.is_test);
    Xte   = X(s.is_test, :);   yte   = y(s.is_test);
    Xtr = Xrest(~s.is_val, :); ytr = yrest(~s.is_val);
    Xva = Xrest(s.is_val, :);  yva = yrest(s.is_val);

    scaler = fit_standardizer(Xtr);              % fitted on train only
    Xtr = apply_standardizer(scaler, Xtr);
    Xva = apply_standardizer(scaler, Xva);
    Xte = apply_standardizer(scaler, Xte);

    prob = make_problem(Xtr, ytr, Xva, yva, cfg.hidden, cfg.lambda);

    o.hist = cell(A, 1);
    o.p_test = cell(A, 1);
    o.y_test = yte;
    o.rows = cell(A, 11);
    for a = 1:A
        rng(cfg.seed * 1000 + k * 10 + a);      % reproducible per run
        [w, h, secs] = train_one(algos{a}, s.w0, prob, cfg, k);
        mtr = binary_metrics(ytr, mlp_forward(w, Xtr, prob.H));
        pte = mlp_forward(w, Xte, prob.H);
        mte = binary_metrics(yte, pte);

        o.hist{a} = h;
        o.p_test{a} = pte;
        o.rows(a, :) = {algos{a}, k, mtr.accuracy, mte.accuracy, ...
            mte.auc, mte.f1, mte.brier, mte.logloss, ...
            h.n_iter, h.best_iter, secs};
        fprintf('  fold %d  %-7s test acc %6.2f %%  AUC %.4f  iters %4d (best %4d)  %7.1f s\n', ...
                k, algos{a}, mte.accuracy, mte.auc, h.n_iter, h.best_iter, secs);
    end
end

function [w, h, secs] = train_one(name, w0, prob, cfg, k)
    patience = cfg.patience;
    if strcmp(name, 'ABC'), patience = cfg.abc.patience; end
    es = early_stopper(patience, cfg.min_delta);
    t = tic;
    switch name
        case 'GD',     [w, h] = train_gd(w0, prob, cfg.gd, es);
        case 'BFGS',   [w, h] = train_bfgs(w0, prob, cfg.bfgs, es);
        case 'L-BFGS', [w, h] = train_lbfgs(w0, prob, cfg.lbfgs, es);
        case 'LM',     [w, h] = train_lm(w0, prob, cfg.lm, es);
        case 'ABC'
            opt = cfg.abc;
            opt.tag = sprintf('fold %d ABC', k);
            [w, h] = train_abc(w0, prob, opt, es);
        otherwise, error('run_experiment:algo', 'Unknown algorithm: %s', name);
    end
    secs = toc(t);
end

function S = summarize(T, algos)
    A = numel(algos);
    vars = {'TrainAcc', 'TestAcc', 'TestAUC', 'TestF1', 'TestBrier', 'Seconds'};
    S = table('Size', [A, 1 + 2 * numel(vars)], ...
              'VariableTypes', [{'string'}, repmat({'double'}, 1, 2 * numel(vars))], ...
              'VariableNames', [{'Algorithm'}, ...
                  reshape([strcat(vars, '_mean'); strcat(vars, '_std')], 1, [])]);
    for a = 1:A
        sel = strcmp(T.Algorithm, algos{a});
        S.Algorithm(a) = algos{a};
        for v = 1:numel(vars)
            x = T.(vars{v})(sel);
            S.([vars{v} '_mean'])(a) = mean(x);
            S.([vars{v} '_std'])(a)  = std(x);
        end
    end
end
