function cfg = default_config()
%DEFAULT_CONFIG All experiment settings in one place.

    % --- Experiment ---
    cfg.seed        = 42;
    cfg.k_folds     = 5;      % outer stratified CV (test folds)
    cfg.val_ratio   = 0.15;   % inner stratified hold-out for early stopping
    cfg.algorithms  = {'GD', 'BFGS', 'L-BFGS', 'LM', 'ABC'};

    % --- Network: 10 inputs -> hidden tanh -> 1 sigmoid output ---
    cfg.hidden      = 32;
    cfg.lambda      = 1e-4;   % L2 penalty on weights (not biases)

    % --- Early stopping on validation loss (shared by all methods) ---
    cfg.patience    = 30;     % iterations / cycles without improvement
    cfg.min_delta   = 1e-5;

    % --- Gradient descent (full batch, classical momentum) ---
    cfg.gd.max_iter = 3000;
    cfg.gd.lr       = 0.5;
    cfg.gd.momentum = 0.9;

    % --- BFGS (dense inverse-Hessian, Armijo line search) ---
    cfg.bfgs.max_iter = 500;

    % --- L-BFGS (two-loop recursion, Armijo line search) ---
    cfg.lbfgs.max_iter = 500;
    cfg.lbfgs.memory   = 10;

    % --- Levenberg-Marquardt (Gauss-Newton matrix J'*S*J) ---
    cfg.lm.max_iter = 200;
    cfg.lm.mu0      = 1e-2;
    cfg.lm.mu_inc   = 10;
    cfg.lm.mu_dec   = 0.1;
    cfg.lm.mu_max   = 1e10;

    % --- Artificial Bee Colony (Karaboga 2005) ---
    cfg.abc.max_cycle   = 1500;
    cfg.abc.food_number = 30;     % employed bees = onlooker bees
    cfg.abc.limit       = 100;    % trials before a source is abandoned
    cfg.abc.mod_rate    = 0.02;   % fraction of weights changed per move
    cfg.abc.bound       = 3;      % weights are kept in [-bound, bound]
    cfg.abc.patience    = 150;    % ABC improves slowly, needs a longer window
end
