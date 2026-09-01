function make_subset_mat(inFile, subsetCsv, outFile)
%MAKE_SUBSET_MAT  Write a new signal .mat holding only the records in the subset.
%
%   make_subset_mat
%   make_subset_mat(INFILE, SUBSETCSV, OUTFILE)
%
%   Defaults, relative to this file:
%       INFILE     SimulatedECGs_Smith2026.mat
%       SUBSETCSV  smith2026_subset.csv
%       OUTFILE    SimulatedECGs_Smith2026_100.mat
%
%   The input is a multi-record file whose variables are cell arrays of ECG
%   tables, one cell array per disease class, and a record is named
%   <Class>_NNN from its position in that cell array. Keeping the first N cells
%   of each class therefore preserves every record name exactly, so labels,
%   signal exports and the corrections CSV all still line up.
%
%   INFILE IS NOT MODIFIED. It is the only copy of these waveforms, it takes
%   seven megabytes, and a record dropped today may be wanted for a robustness
%   check tomorrow. Move it aside by hand once you are satisfied with the output.
%
%   The subset is read from SUBSETCSV rather than hard-coded, so the file on disk
%   remains the single statement of which records the study uses.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(inFile);    inFile    = fullfile(here,'SimulatedECGs_Smith2026.mat'); end
    if nargin < 2 || isempty(subsetCsv); subsetCsv = fullfile(here,'smith2026_subset.csv'); end
    if nargin < 3 || isempty(outFile);   outFile   = fullfile(here,'SimulatedECGs_Smith2026_100.mat'); end
    if exist(inFile,'file')    ~= 2; error('subset:in','Cannot find %s', inFile); end
    if exist(subsetCsv,'file') ~= 2; error('subset:csv','Cannot find %s', subsetCsv); end

    W = readtable(subsetCsv,'TextType','string');
    if ~ismember('record_id', W.Properties.VariableNames)
        error('subset:cols','%s has no record_id column.', subsetCsv);
    end
    want = containers.Map('KeyType','char','ValueType','logical');
    for i = 1:height(W); want(strtrim(char(W.record_id(i)))) = true; end
    fprintf('Subset names %d records.\n', want.Count);

    raw = load(inFile);
    fn  = fieldnames(raw);
    out = struct();
    kept = 0; dropped = 0; keptNames = {};

    for i = 1:numel(fn)
        v   = raw.(fn{i});
        cls = regexprep(char(fn{i}), 'ECGs?$', '');
        if ~iscell(v)
            out.(fn{i}) = v;                 % carry anything that is not a class
            continue;
        end
        sel = {};
        for k = 1:numel(v)
            nm = sprintf('%s_%03d', cls, k);
            if isKey(want, nm)
                sel{end+1} = v{k}; %#ok<AGROW>
                keptNames{end+1} = nm; %#ok<AGROW>
                kept = kept + 1;
            else
                dropped = dropped + 1;
            end
        end
        % Reshape to the input's own orientation so downstream code that expects
        % a row or column cell array is not surprised by the subset.
        if size(v,1) == 1; sel = reshape(sel, 1, []); else; sel = reshape(sel, [], 1); end
        out.(fn{i}) = sel;
        fprintf('  %-22s kept %3d of %3d\n', cls, numel(sel), numel(v));
    end

    missing = setdiff(keys(want), keptNames);
    if ~isempty(missing)
        error('subset:missing', ...
            ['%d record(s) named in the subset are not in %s, first is %s.\n' ...
             'Nothing was written. Fix the subset file and run again.'], ...
            numel(missing), inFile, missing{1});
    end

    % Already built. Returning rather than erroring, so pasting the whole setup
    % block a second time does not abort the lines that follow it.
    if exist(outFile,'file') == 2
        fprintf('%s already exists, so nothing was rebuilt.\n', outFile);
        fprintf('Delete or rename it first if you really do want it rebuilt.\n');
        return;
    end
    save(outFile, '-struct', 'out', '-v7');
    fprintf('\nWrote %s\n  %d records kept, %d dropped.\n', outFile, kept, dropped);
    fprintf('  %s is unchanged. Move it aside by hand when you are ready.\n', inFile);

    % A record whose numbering is not contiguous would silently rename records on
    % reload, since the tool names them from cell position. Checking here is
    % cheap and the failure would otherwise appear as labels attached to the
    % wrong waveform.
    chk = load(outFile);
    cfn = fieldnames(chk); reNames = {};
    for i = 1:numel(cfn)
        v = chk.(cfn{i}); if ~iscell(v); continue; end
        c = regexprep(char(cfn{i}), 'ECGs?$', '');
        for k = 1:numel(v); reNames{end+1} = sprintf('%s_%03d', c, k); end %#ok<AGROW>
    end
    bad = setdiff(keptNames, reNames);
    if isempty(bad)
        fprintf('  Verified, every kept record reloads under its original name.\n');
    else
        warning(['%d record(s) would reload under a DIFFERENT name, first is %s. ' ...
                 'That happens when the kept records are not 001..N of their class. ' ...
                 'Do not use this file.'], numel(bad), bad{1});
    end
end
