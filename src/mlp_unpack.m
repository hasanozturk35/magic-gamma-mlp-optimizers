function [W1, b1, W2, b2] = mlp_unpack(w, D, H)
%MLP_UNPACK Inverse of the packing used in MLP_INIT.

    expected = H*D + H + H + 1;
    if numel(w) ~= expected
        error('mlp_unpack:size', 'Expected %d parameters, got %d.', ...
              expected, numel(w));
    end
    W1 = reshape(w(1:H*D), H, D);
    b1 = w(H*D+1 : H*D+H).';
    W2 = w(H*D+H+1 : H*D+2*H).';
    b2 = w(end);
end
