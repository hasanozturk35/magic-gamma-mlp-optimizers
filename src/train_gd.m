function [w_best, hist] = train_gd(w0, prob, opt, es)
%TRAIN_GD Full-batch gradient descent with classical (heavy-ball) momentum.

    w = w0;
    v = zeros(size(w));
    hist = init_history(opt.max_iter);

    for it = 1:opt.max_iter
        [f, g] = prob.fobj(w);
        v = opt.momentum * v - opt.lr * g;
        w = w + v;

        hist.train_loss(it) = f;
        hist.val_loss(it)   = prob.val_loss(w);
        if es.update(hist.val_loss(it), w, it), break; end
    end

    [w_best, hist] = finish_history(hist, it, es, w);
end
