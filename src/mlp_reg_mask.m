function m = mlp_reg_mask(D, H)
%MLP_REG_MASK 1 for weights, 0 for biases (L2 is applied to weights only).

    m = [ones(H*D, 1); zeros(H, 1); ones(H, 1); 0];
end
