function Xs = apply_standardizer(scaler, X)
%APPLY_STANDARDIZER Apply statistics from FIT_STANDARDIZER to any split.

    Xs = (X - scaler.mu) ./ scaler.sigma;
end
