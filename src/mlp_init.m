function w = mlp_init(D, H)
%MLP_INIT Glorot/Xavier-normal weights, zero biases, packed into a column.
%   Packing order: [W1(:); b1(:); W2(:); b2]  with W1 (H x D), W2 (1 x H).

    W1 = randn(H, D) * sqrt(2 / (D + H));
    b1 = zeros(H, 1);
    W2 = randn(1, H) * sqrt(2 / (H + 1));
    b2 = 0;
    w = [W1(:); b1; W2(:); b2];
end
