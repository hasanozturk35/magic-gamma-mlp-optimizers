function scaler = fit_standardizer(X)
%FIT_STANDARDIZER Z-score statistics. Call on TRAINING data only.

    scaler.mu    = mean(X, 1);
    scaler.sigma = std(X, 0, 1);
    scaler.sigma(scaler.sigma < eps) = 1;   % constant feature: avoid /0
end
