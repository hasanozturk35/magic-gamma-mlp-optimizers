classdef early_stopper < handle
%EARLY_STOPPER Track validation loss, keep the best weights, signal stop.
%   es = early_stopper(patience, min_delta);
%   stop = es.update(val_loss, w, iter);
%   es.best_w / es.best_iter / es.best_loss

    properties
        patience
        min_delta
        best_loss = inf
        best_w    = []
        best_iter = 0
        stall     = 0
    end

    methods
        function obj = early_stopper(patience, min_delta)
            obj.patience  = patience;
            obj.min_delta = min_delta;
        end

        function stop = update(obj, val_loss, w, iter)
            if val_loss < obj.best_loss - obj.min_delta
                obj.best_loss = val_loss;
                obj.best_w    = w;
                obj.best_iter = iter;
                obj.stall     = 0;
            else
                obj.stall = obj.stall + 1;
            end
            stop = obj.stall >= obj.patience;
        end
    end
end
