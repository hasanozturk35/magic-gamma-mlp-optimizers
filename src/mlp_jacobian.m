function [J, p] = mlp_jacobian(w, X, H)
%MLP_JACOBIAN Jacobian of the output logit z w.r.t. all parameters (N x P).
%   Column order matches the packing [W1(:); b1; W2(:); b2].

    [N, D] = size(X);
    [~, ~, W2, ~] = mlp_unpack(w, D, H);
    [p, ~, A1] = mlp_forward(w, X, H);

    G   = (1 - A1.^2) .* W2;                                   % dz/db1, N x H
    JW1 = reshape(G .* reshape(X, N, 1, D), N, H*D);           % dz/dW1(h,d)
    J   = [JW1, G, A1, ones(N, 1)];
end
