function [loss, grad] = mlp_loss_grad(w, X, y, H, lambda)
%MLP_LOSS_GRAD Mean binary cross-entropy + (lambda/2)*||weights||^2.
%   The cross-entropy is computed from the logit with a numerically stable
%   softplus, so it never takes log(0).

    if nargin < 5, lambda = 0; end
    [N, D] = size(X);
    [W1, ~, W2, ~] = mlp_unpack(w, D, H);
    [p, z, A1] = mlp_forward(w, X, H);

    softplus = max(z, 0) + log1p(exp(-abs(z)));     % log(1 + e^z)
    loss = mean(softplus - y .* z) ...
         + (lambda / 2) * (sum(W1(:).^2) + sum(W2.^2));

    if nargout > 1
        dz = (p - y) / N;                            % N x 1
        gW2 = dz.' * A1 + lambda * W2;               % 1 x H
        gb2 = sum(dz);
        dA  = (dz * W2) .* (1 - A1.^2);              % N x H
        gW1 = dA.' * X + lambda * W1;                % H x D
        gb1 = sum(dA, 1).';
        grad = [gW1(:); gb1; gW2(:); gb2];
    end
end
