function [X, y, feature_names] = load_magic04(csv_path)
%LOAD_MAGIC04 Read and validate the UCI MAGIC Gamma Telescope dataset.
%   y = 1 for gamma (signal, 'g'), y = 0 for hadron (background, 'h').

    feature_names = {'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', ...
                     'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'};

    if ~isfile(csv_path)
        error('load_magic04:missing', ...
              'Dataset not found: %s (see data/README.md)', csv_path);
    end

    opts = delimitedTextImportOptions('NumVariables', 11, 'Delimiter', ',');
    opts.VariableNames = [feature_names, {'class'}];
    opts.VariableTypes = [repmat({'double'}, 1, 10), {'string'}];
    opts.ExtraColumnsRule = 'error';
    opts.EmptyLineRule    = 'skip';
    tbl = readtable(csv_path, opts);

    X = tbl{:, 1:10};
    labels = strtrim(tbl.class);

    if any(~isfinite(X(:)))
        error('load_magic04:badValue', 'Non-numeric or missing feature values.');
    end
    if ~all(ismember(labels, ["g", "h"]))
        error('load_magic04:badLabel', 'Labels must be ''g'' or ''h''.');
    end

    y = double(labels == "g");
end
