function tests = test_core
%TEST_CORE Unit tests. Run from the repository root:
%   >> results = runtests('tests');

    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = fileparts(fileparts(mfilename('fullpath')));
    tc.TestData.root = root;
    addpath(fullfile(root, 'src'));
end

% ---------------------------------------------------------------- network

function test_gradient_matches_finite_differences(tc)
    rng(1);
    D = 4; H = 5; N = 30;
    X = randn(N, D); y = double(rand(N, 1) > 0.5);
    w = mlp_init(D, H) + 0.1 * randn(H*D + 2*H + 1, 1);
    lambda = 0.01;
    [~, g] = mlp_loss_grad(w, X, y, H, lambda);
    g_num = zeros(size(w)); e = 1e-6;
    for i = 1:numel(w)
        d = zeros(size(w)); d(i) = e;
        g_num(i) = (mlp_loss_grad(w + d, X, y, H, lambda) ...
                  - mlp_loss_grad(w - d, X, y, H, lambda)) / (2 * e);
    end
    tc.verifyLessThan(norm(g - g_num) / norm(g_num), 1e-6);
end

function test_jacobian_matches_finite_differences(tc)
    rng(2);
    D = 3; H = 4; N = 12;
    X = randn(N, D);
    w = mlp_init(D, H) + 0.1 * randn(H*D + 2*H + 1, 1);
    J = mlp_jacobian(w, X, H);
    J_num = zeros(size(J)); e = 1e-6;
    for i = 1:numel(w)
        d = zeros(size(w)); d(i) = e;
        [~, zp] = mlp_forward(w + d, X, H);
        [~, zm] = mlp_forward(w - d, X, H);
        J_num(:, i) = (zp - zm) / (2 * e);
    end
    tc.verifyLessThan(max(abs(J(:) - J_num(:))), 1e-7);
end

function test_loss_is_stable_for_large_logits(tc)
    D = 1; H = 1;
    w = [100; 0; 100; 0];                  % huge, saturated logits
    X = [1; -1]; y = [0; 1];               % both confidently wrong
    L = mlp_loss_grad(w, X, y, H, 0);
    tc.verifyTrue(isfinite(L));
    tc.verifyGreaterThan(L, 50);
end

function test_unpack_rejects_wrong_size(tc)
    tc.verifyError(@() mlp_unpack(zeros(5, 1), 3, 4), 'mlp_unpack:size');
end

% ---------------------------------------------------------------- data split

function test_stratified_kfold_is_a_partition_and_balanced(tc)
    rng(3);
    y = [ones(650, 1); zeros(350, 1)];
    K = 5;
    fold = stratified_kfold(y, K);
    tc.verifyEqual(sort(unique(fold)).', 1:K);      % every sample in one fold
    for k = 1:K
        tc.verifyEqual(sum(fold == k), 200, 'AbsTol', 1);
        tc.verifyEqual(mean(y(fold == k)), 0.65, 'AbsTol', 0.01);
    end
end

function test_holdout_is_stratified(tc)
    rng(4);
    y = [ones(700, 1); zeros(300, 1)];
    v = stratified_holdout(y, 0.2);
    tc.verifyEqual(sum(v), 200);
    tc.verifyEqual(mean(y(v)), 0.7, 'AbsTol', 1e-12);
end

function test_standardizer_uses_only_fit_data(tc)
    Xtr = [1 10; 3 10; 5 10];
    Xte = [1000 -5];
    s = fit_standardizer(Xtr);
    tc.verifyEqual(s.mu, [3 10]);
    tc.verifyEqual(s.sigma, [2 1]);                 % constant column -> 1
    tc.verifyEqual(apply_standardizer(s, Xte), [498.5 -15]);
end

% ---------------------------------------------------------------- metrics

function test_auc_known_values(tc)
    m = binary_metrics([0 0 1 1], [0.1 0.4 0.35 0.8]);
    tc.verifyEqual(m.auc, 0.75, 'AbsTol', 1e-12);
    m = binary_metrics([0 1 0 1], [0.5 0.5 0.5 0.5]);  % all tied -> 0.5
    tc.verifyEqual(m.auc, 0.5, 'AbsTol', 1e-12);
    m = binary_metrics([0 0 1 1], [0.1 0.2 0.8 0.9]);
    tc.verifyEqual(m.auc, 1);
    tc.verifyEqual(m.accuracy, 100);
    tc.verifyEqual(m.f1, 1);
end

function test_roc_endpoints(tc)
    [fpr, tpr] = roc_curve([0 1 0 1 1], [0.2 0.9 0.6 0.4 0.7]);
    tc.verifyEqual([fpr(1) tpr(1)], [0 0]);
    tc.verifyEqual([fpr(end) tpr(end)], [1 1]);
    tc.verifyTrue(all(diff(fpr) >= 0) && all(diff(tpr) >= 0));
end

% ---------------------------------------------------------------- loading

function test_loader_reads_real_dataset(tc)
    [X, y] = load_magic04(fullfile(tc.TestData.root, 'data', 'magic04.csv'));
    tc.verifySize(X, [19020 10]);
    tc.verifyEqual(sum(y), 12332);
end

function test_loader_rejects_bad_label(tc)
    f = [tempname '.csv'];
    c = onCleanup(@() delete(f));
    fid = fopen(f, 'w');
    fprintf(fid, '1,2,3,4,5,6,7,8,9,10,g\n1,2,3,4,5,6,7,8,9,10,x\n');
    fclose(fid);
    tc.verifyError(@() load_magic04(f), 'load_magic04:badLabel');
end

function test_loader_rejects_missing_value(tc)
    f = [tempname '.csv'];
    c = onCleanup(@() delete(f));
    fid = fopen(f, 'w');
    fprintf(fid, '1,2,3,4,5,6,7,8,9,10,g\n1,2,,4,5,6,7,8,9,10,h\n');
    fclose(fid);
    tc.verifyError(@() load_magic04(f), 'load_magic04:badValue');
end

function test_loader_rejects_missing_file(tc)
    tc.verifyError(@() load_magic04('does_not_exist.csv'), 'load_magic04:missing');
end

% ---------------------------------------------------------------- optimizers

function test_every_optimizer_reduces_loss(tc)
    rng(5);
    N = 400; D = 4; H = 6;
    X = randn(N, D);
    y = double(X(:, 1) + 0.5 * X(:, 2).^2 - 0.5 > 0);
    va = 1:100; tr = 101:N;
    prob = make_problem(X(tr, :), y(tr), X(va, :), y(va), H, 1e-4);
    w0 = mlp_init(D, H);
    L0 = prob.val_loss(w0);

    cfg = default_config();
    cfg.gd.max_iter = 300; cfg.bfgs.max_iter = 100; cfg.lbfgs.max_iter = 100;
    cfg.lm.max_iter = 50;  cfg.abc.max_cycle = 150;
    trainers = {@(es) train_gd(w0, prob, cfg.gd, es), ...
                @(es) train_bfgs(w0, prob, cfg.bfgs, es), ...
                @(es) train_lbfgs(w0, prob, cfg.lbfgs, es), ...
                @(es) train_lm(w0, prob, cfg.lm, es), ...
                @(es) train_abc(w0, prob, cfg.abc, es)};
    names = {'GD', 'BFGS', 'L-BFGS', 'LM', 'ABC'};
    for t = 1:numel(trainers)
        es = early_stopper(20, 1e-6);
        [w, h] = trainers{t}(es);
        tc.verifyLessThan(prob.val_loss(w), 0.9 * L0, names{t});
        tc.verifyEqual(prob.val_loss(w), min(h.val_loss), 'AbsTol', 1e-6, ...
                       [names{t} ' must return best-validation weights']);
    end
end
