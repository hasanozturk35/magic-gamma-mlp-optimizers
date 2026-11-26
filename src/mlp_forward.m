function [p, z, A1] = mlp_forward(w, X, H)
%MLP_FORWARD One hidden tanh layer, sigmoid output.
%   p : P(y = 1 | x)   (N x 1)
%   z : output logit   (N x 1)
%   A1: hidden activations (N x H)

    D = size(X, 2);
    [W1, b1, W2, b2] = mlp_unpack(w, D, H);
    A1 = tanh(X * W1.' + b1);
    z  = A1 * W2.' + b2;
    p  = 1 ./ (1 + exp(-z));
end
