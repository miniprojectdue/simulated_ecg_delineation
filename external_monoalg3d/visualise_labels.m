function visualise_labels(recPath, labelPath)
%VISUALISE_LABELS  12-lead overview of an internal ECG record with its
%   saved manual fiducial labels.
%
%   visualise_labels                       % pick record + labels via dialogs
%   visualise_labels(RECORD_MAT)           % auto-find the label file
%   visualise_labels(RECORD_MAT, LABELS)   % explicit labels .csv or .mat
%
%   RECORD_MAT : an ecg_table-style record (table OR struct of columns) with
%                a time column (Time/t) and I II III aVR aVL aVF V1..V6.
%   LABELS     : the *_labels.csv (preferred) or *_labels.mat written by
%                manual_label_ecg. If omitted, the tool looks for
%                <record>_labels.csv (then .mat) next to the record and in
%                a ./labels subfolder.
%
%   The plot ADAPTS to whichever fiducials are present in the label file
%   (the 5-point QRS/T set, the 6-point set that also has R peak, or the
%   8-point set that adds Q and S), so it never assumes a fixed count. A
%   landmark saved as blank/NaN is genuinely absent on that lead and is not
%   drawn. Leads flagged "exclude" are greyed and an "inverted T" flag is
%   noted in the sub-title.
%
%   Reading the CSV is preferred because its named columns state exactly
%   which fiducials were saved; the .mat carries the same points.

REG    = fiducialRegistry();                 % known fiducials: key/label/color
LEADS  = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
SUBPOS = [1 5 9  2 6 10  3 7 11  4 8 12];    % clinical 3x4 layout

% -------------------------------------------------------- resolve inputs
if nargin < 1 || isempty(recPath)
    [fn,fp] = uigetfile({'*.mat','ECG record (*.mat)'},'Select the ECG record');
    if isequal(fn,0); disp('Cancelled.'); return; end
    recPath = fullfile(fp,fn);
end
[recDir, recName] = fileparts(recPath);

if nargin < 2 || isempty(labelPath)
    cand = { fullfile(recDir,[recName '_labels.csv']), ...
             fullfile(recDir,'labels',[recName '_labels.csv']), ...
             fullfile(recDir,[recName '_labels.mat']), ...
             fullfile(recDir,'labels',[recName '_labels.mat']) };
    labelPath = '';
    for c = 1:numel(cand)
        if exist(cand{c},'file')==2; labelPath = cand{c}; break; end
    end
    if isempty(labelPath)
        [fn,fp] = uigetfile({'*.csv;*.mat','Label file'}, ...
            sprintf('Select labels for %s',recName));
        if isequal(fn,0); disp('Cancelled.'); return; end
        labelPath = fullfile(fp,fn);
    end
end
fprintf('Record : %s\nLabels : %s\n', recPath, labelPath);

% ------------------------------------------------------------- load data
raw = load(recPath);
tbl = pickTable(raw);
if isempty(tbl); error('No ECG table/struct found in %s.', recPath); end

vn = tbl.Properties.VariableNames;
if any(strcmp(vn,'Time')); tcol = double(tbl.Time);
elseif any(strcmp(vn,'t')); tcol = double(tbl.t);
else; tcol = (0:height(tbl)-1)'; end
t = tcol - tcol(1);

[L, fkeys] = loadLabels(labelPath, LEADS, {REG.key});
fprintf('Fiducials in file: %s\n', strjoin(fkeys, ', '));

% --------------------------------------------------------------- plot
figure('Name',['Labels: ' recName],'NumberTitle','off','Color','w', ...
    'Units','normalized','Position',[0.05 0.08 0.9 0.82]);

for i = 1:numel(LEADS)
    lead = LEADS{i};
    if ~any(strcmp(vn,lead)); continue; end
    V  = double(tbl.(lead));
    ax = subplot(3,4,SUBPOS(i)); hold(ax,'on'); grid(ax,'on');

    ent = L(i);
    traceCol = [0 0 0];
    if ent.exclude; traceCol = [0.65 0.65 0.65]; end
    plot(ax, t, V, '-', 'Color',traceCol,'LineWidth',1.1);

    idx = clampVec(ent.idx, numel(t));
    for k = 1:numel(fkeys)
        if ~isfinite(idx(k)); continue; end     % landmark absent on this lead
        plot(ax, t(idx(k)), V(idx(k)), '.', 'MarkerSize',16, ...
            'Color', colorFor(fkeys{k}, REG));
    end

    ttl = lead;
    if ent.invertedT; ttl = [lead '  (inv T)'];   end
    if ent.exclude;   ttl = [lead '  (excluded)']; end
    title(ax, ttl);
    xlim(ax,[t(1) t(end)]);
end

sgtitle(strrep(recName,'_','\_'), 'FontSize',13,'FontWeight','bold');

% single shared legend, built from the fiducials actually present
lh = gobjects(1,numel(fkeys)); lbl = cell(1,numel(fkeys));
hold on
for k = 1:numel(fkeys)
    lh(k)  = plot(NaN,NaN,'.','MarkerSize',18,'Color',colorFor(fkeys{k},REG));
    lbl{k} = labelFor(fkeys{k}, REG);
end
legend(lh, lbl, 'Orientation','horizontal', ...
    'Position',[0.20 0.005 0.6 0.03]);
end

% ======================================================================
function REG = fiducialRegistry()
%FIDUCIALREGISTRY  Master ordered list of known fiducials with plot styling.
    keys = {'QRS_start','Q_peak','R_peak','S_peak','QRS_end', ...
            'T_start','T_peak','T_end'};
    labs = {'QRS start','Q','R','S','QRS end','T start','T peak','T end'};
    cols = [0.85 0.10 0.10; 0.00 0.45 0.45; 0.90 0.10 0.55; ...
            0.55 0.35 0.10; 0.90 0.45 0.00; ...
            0.10 0.55 0.85; 0.20 0.65 0.20; 0.55 0.20 0.70];
    REG = struct('key',keys,'label',labs);
    for i = 1:numel(keys); REG(i).color = cols(i,:); end
end

function c = colorFor(key, REG)
    j = find(strcmp({REG.key}, key),1);
    if isempty(j); c = [0.3 0.3 0.3]; else; c = REG(j).color; end
end

function s = labelFor(key, REG)
    j = find(strcmp({REG.key}, key),1);
    if isempty(j); s = strrep(key,'_',' '); else; s = REG(j).label; end
end

function L = initL(leads, nF)
    nL = numel(leads);
    L = repmat(struct('lead','','idx',nan(1,nF), ...
        'invertedT',false,'exclude',false), 1, nL);
    for i = 1:nL; L(i).lead = leads{i}; end
end

function [L, fkeys] = loadLabels(labelPath, leads, masterKeys)
%LOADLABELS  Read a *_labels.csv or *_labels.mat and return per-lead labels
%   plus the ordered list of fiducial keys actually present in the file.
    nL = numel(leads);
    fkeys = {};
    [~,~,ext] = fileparts(labelPath);

    if strcmpi(ext,'.mat')
        s = load(labelPath);
        if isfield(s,'L')
            if isfield(s,'meta') && isfield(s.meta,'fids') && ~isempty(s.meta.fids)
                fkeys = cellstr(s.meta.fids);           % names as saved
            else
                m = numel(s.L(1).idx);                  % infer from width
                fkeys = masterKeys(1:min(m,numel(masterKeys)));
            end
            L = initL(leads, numel(fkeys));
            for i = 1:min(nL,numel(s.L))
                j = find(strcmp({L.lead}, s.L(i).lead),1);
                if isempty(j); j = i; end
                id = s.L(i).idx(:).';
                L(j).idx       = id(1:numel(fkeys));
                L(j).invertedT = logical(s.L(i).invertedT);
                L(j).exclude   = logical(s.L(i).exclude);
            end
            return
        end
    end

    % CSV: fiducials present = master keys that have a <key>_idx column
    T  = readtable(labelPath);
    vn = T.Properties.VariableNames;
    for k = 1:numel(masterKeys)
        if ismember([masterKeys{k} '_idx'], vn); fkeys{end+1} = masterKeys{k}; end %#ok<AGROW>
    end
    L = initL(leads, numel(fkeys));
    for i = 1:height(T)
        leadName = char(string(T.Lead(i)));
        j = find(strcmp({L.lead}, leadName),1);
        if isempty(j); continue; end
        id = zeros(1,numel(fkeys));
        for k = 1:numel(fkeys); id(k) = T.([fkeys{k} '_idx'])(i); end
        L(j).idx = id;
        if ismember('invertedT',vn); L(j).invertedT = logical(T.invertedT(i)); end
        if ismember('exclude',vn);   L(j).exclude   = logical(T.exclude(i));   end
    end
end

% ---------------------------------------------------- shared table helpers
function tbl = pickTable(raw)
%PICKTABLE  Return an ECG record as a table from a loaded .mat, accepting a
%   table variable, a scalar struct of columns, or loose column variables.
    tbl = [];
    fn = fieldnames(raw);
    for i = 1:numel(fn)
        if istable(raw.(fn{i})); tbl = raw.(fn{i}); return; end
    end
    for i = 1:numel(fn)
        v = raw.(fn{i});
        if isstruct(v) && isscalar(v) && looksLikeEcg(fieldnames(v))
            tbl = buildTableFromStruct(v);
            if ~isempty(tbl); return; end
        end
    end
    if looksLikeEcg(fn); tbl = buildTableFromStruct(raw); end
end

function tf = looksLikeEcg(names)
    leads = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    nLead = sum(ismember(leads, names));
    hasTime = any(ismember({'Time','t'}, names));
    tf = (nLead >= 6) && (hasTime || nLead >= 8);
end

function tbl = buildTableFromStruct(s)
    want = {'Time','t','I','II','III','aVR','aVL','aVF', ...
            'V1','V2','V3','V4','V5','V6'};
    names = fieldnames(s);
    keep = {}; cols = {}; N = [];
    for i = 1:numel(want)
        nm = want{i};
        if any(strcmp(names, nm))
            v = s.(nm);
            if isnumeric(v) && isvector(v)
                v = double(v(:));
                if isempty(N); N = numel(v); end
                if numel(v) == N
                    keep{end+1} = nm; %#ok<AGROW>
                    cols{end+1} = v;  %#ok<AGROW>
                end
            end
        end
    end
    if numel(keep) < 2; tbl = []; else; tbl = table(cols{:}, 'VariableNames', keep); end
end

function idx = clampVec(idx, n)
%CLAMPVEC  Round and bound the finite entries, leaving absent landmarks NaN
%   so the caller can skip them rather than draw a marker at sample 1.
    ok = isfinite(idx);
    idx(ok)  = max(1, min(n, round(idx(ok))));
    idx(~ok) = NaN;
end
