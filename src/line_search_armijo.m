function [w_new, f_new, g_new, alpha] = line_search_armijo(fobj, w, f, g, p, alpha0)
%LINE_SEARCH_ARMIJO Backtracking line search (sufficient-decrease condition).
%   fobj(w) must return [f, g]. p must be a descent direction (g'*p < 0).
%   If no acceptable step is found, returns w unchanged (alpha = 0).

    c1 = 1e-4;
    shrink = 0.5;
    max_steps = 30;

    slope = g.' * p;
    alpha = alpha0;
    for k = 1:max_steps
        w_try = w + alpha * p;
        [f_try, g_try] = fobj(w_try);
        if isfinite(f_try) && f_try <= f + c1 * alpha * slope
            w_new = w_try; f_new = f_try; g_new = g_try;
            return;
        end
        alpha = alpha * shrink;
    end
    w_new = w; f_new = f; g_new = g; alpha = 0;
end
