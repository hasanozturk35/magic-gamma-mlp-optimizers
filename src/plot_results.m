function plot_results(R, out_dir)
%PLOT_RESULTS Save comparison figures as PNG into out_dir.

    algos = R.algorithms;
    A = numel(algos);
    T = R.per_fold;
    K = max(T.Fold);
    has_curves = isfield(R, 'hist') && ~isempty(R.hist) && ~isempty(R.hist{1});
    S = R.summary;
    colors = lines(A);
    vis = 'on';
    if ~usejava('desktop'), vis = 'off'; end

    % --- 1. accuracy & AUC (mean +- std, with per-fold points)
    f = new_fig([100 100 1200 420], vis);
    metrics = {'TestAcc', 'Test accuracy (%)'; 'TestAUC', 'Test ROC AUC'};
    for m = 1:2
        subplot(1, 2, m); hold on;
        mu = S.([metrics{m, 1} '_mean']);
        sd = S.([metrics{m, 1} '_std']);
        b = bar(1:A, mu, 0.6, 'FaceColor', 'flat');
        b.CData = colors;
        errorbar(1:A, mu, sd, 'k.', 'LineWidth', 1.2);
        for a = 1:A
            x = T.(metrics{m, 1})(strcmp(T.Algorithm, algos{a}));
            scatter(a + 0.12 * (rand(size(x)) - 0.5), x, 18, 'k', 'filled', ...
                    'MarkerFaceAlpha', 0.5);
        end
        lo = min(T.(metrics{m, 1})); hi = max(T.(metrics{m, 1}));
        pad = 0.15 * (hi - lo + eps);
        ylim([lo - 3 * pad, hi + pad]);
        set(gca, 'XTick', 1:A, 'XTickLabel', algos);
        ylabel(metrics{m, 2}); grid on; box on;
        title(sprintf('%s (%d-fold CV)', metrics{m, 2}, K));
    end
    save_fig(f, fullfile(out_dir, 'accuracy_auc.png'));

    if has_curves
        plot_curves(R, algos, A, K, colors, vis, out_dir);
    end

    % --- 4. training time
    f = new_fig([100 100 640 420], vis);
    b = bar(1:A, S.Seconds_mean, 0.6, 'FaceColor', 'flat');
    b.CData = colors;
    set(gca, 'XTick', 1:A, 'XTickLabel', algos, 'YScale', 'log');
    t = max(S.Seconds_mean, 1e-3);
    lo = 10^floor(log10(min(t)));
    hi = max(10^ceil(log10(max(t)) + 0.3), 10 * lo);   % room for the labels
    ylim([lo, hi]);
    text(1:A, S.Seconds_mean, compose('%.1f s', S.Seconds_mean), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    ylabel('Seconds per fold (log scale)'); grid on; box on;
    title('Training time');
    save_fig(f, fullfile(out_dir, 'training_time.png'));
end

function plot_curves(R, algos, A, K, colors, vis, out_dir)
    % --- 2. validation loss curves (fold 1 .. K, mean shown bold)
    f = new_fig([100 100 1400 760], vis);
    for a = 1:A
        subplot(2, 3, a); hold on;
        for k = 1:K
            plot(R.hist{a, k}.val_loss, 'Color', [colors(a, :) 0.45]);
        end
        h = R.hist{a, 1};
        plot(h.train_loss, '--', 'Color', [0.3 0.3 0.3]);
        set(gca, 'XScale', 'log'); grid on; box on;
        xlabel('Iteration / cycle'); ylabel('Cross-entropy');
        title(algos{a});
        if a == 1, legend({'val (each fold)'}, 'Location', 'northeast'); end
    end
    subplot(2, 3, 6); hold on;
    for a = 1:A
        L = cellfun(@(h) min([h.val_loss; NaN]), R.hist(a, :));
        plot(a * ones(1, K), L, 'o', 'Color', colors(a, :), 'MarkerFaceColor', colors(a, :));
    end
    set(gca, 'XTick', 1:A, 'XTickLabel', algos); xlim([0.5 A + 0.5]);
    ylabel('Best validation loss'); grid on; box on;
    title('Best validation loss per fold');
    sgtitle('Validation loss (solid) and fold-1 training loss (dashed)');
    save_fig(f, fullfile(out_dir, 'loss_curves.png'));

    % --- 3. pooled ROC curves over all test folds
    f = new_fig([100 100 640 600], vis);
    hold on;
    for a = 1:A
        [fpr, tpr] = roc_curve(R.labels{a}, R.scores{a});
        m = binary_metrics(R.labels{a}, R.scores{a});
        plot(fpr, tpr, 'LineWidth', 2, 'Color', colors(a, :), ...
             'DisplayName', sprintf('%s (AUC %.3f)', algos{a}, m.auc));
    end
    plot([0 1], [0 1], 'k:', 'DisplayName', 'chance');
    axis square; grid on; box on;
    xlabel('False positive rate (hadron accepted as gamma)');
    ylabel('True positive rate (gamma detected)');
    title('ROC, pooled test predictions');
    legend('Location', 'southeast');
    save_fig(f, fullfile(out_dir, 'roc_curves.png'));
end

function f = new_fig(pos, vis)
    f = figure('Color', 'w', 'Position', pos, 'Visible', vis);
    if exist('theme', 'file') || exist('theme', 'builtin')
        theme(f, 'light');        % R2025a+ defaults to the desktop (dark) theme
    end
end

function save_fig(f, path)
    exportgraphics(f, path, 'Resolution', 150);
    if strcmp(f.Visible, 'off'), close(f); end
end
