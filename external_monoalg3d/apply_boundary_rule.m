function apply_boundary_rule(matFile, labelDir, corrCsv, mode)
%APPLY_BOUNDARY_RULE  Set the four beat boundaries of every record from the
%   spatial-magnitude 5 per cent rule, in one pass, instead of by hand.
%
%   apply_boundary_rule
%   apply_boundary_rule(MATFILE, LABELDIR, CORRCSV, MODE)
%
%   Defaults (relative to this file):
%       MATFILE   SimulatedECGs_Smith2026.mat
%       LABELDIR  labels/
%       CORRCSV   labels/smith2026_manual_corrections.csv
%       MODE      'all'
%
%   -- What it changes ---------------------------------------------------
%   QRS onset, QRS offset, T onset and T offset ONLY. Q, R, S and T peaks are
%   left exactly as you placed them, as are the presence flags, the inverted-T
%   and exclude flags, and every v3_* seed column. Peaks are where the eye adds
%   something the magnitude cannot supply, because squaring removes polarity.
%
%   MODE = 'all'      every record, every boundary  (default)
%   MODE = 'defects'  only the boundaries that are demonstrably wrong: T onset
%                     at or before QRS offset, T onset before the magnitude
%                     trough, and QRS onset still sitting on the delineator's
%                     11 ms search-window floor. Everything else keeps your
%                     hand placement.
%
%

%
%   Run apply_boundary_rule then reopen manual_label_ecg to see the result.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(matFile);  matFile  = fullfile(here,'SimulatedECGs_Smith2026.mat'); end
    if nargin < 2 || isempty(labelDir); labelDir = fullfile(here,'labels'); end
    if nargin < 3 || isempty(corrCsv);  corrCsv  = fullfile(labelDir,'smith2026_manual_corrections.csv'); end
    if nargin < 4 || isempty(mode);     mode     = 'all'; end
    mode = lower(mode);
    if ~ismember(mode, {'all','defects'})
        error('apply:mode','MODE must be ''all'' or ''defects''.');
    end
    if exist(matFile,'file') ~= 2; error('apply:mat','Cannot find %s', matFile); end
    if exist(corrCsv,'file') ~= 2; error('apply:csv','Cannot find %s', corrCsv); end

    LEADS    = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    MAGLEADS = {'I','II','V1','V2','V3','V4','V5','V6'};
    stamp    = datestr(now,'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>

    fprintf('\napply_boundary_rule  (mode: %s)\n', mode);
    fprintf('  data   %s\n  labels %s\n  csv    %s\n\n', matFile, labelDir, corrCsv);

    % ---------------------------------------------------- reference values
    sig = loadSignals(matFile, LEADS);
    ids = sig.keys();
    B   = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(ids)
        s = sig(ids{i});
        b = ruleBounds(s.M, LEADS, MAGLEADS);
        if ~isempty(b); B(ids{i}) = b; end
    end
    fprintf('reference boundaries computed for %d of %d records\n\n', B.Count, numel(ids));

    % ---------------------------------------------------- rewrite the CSV
    txt = fileread(corrCsv);
    lines = regexp(txt, '\r\n|\n|\r', 'split');
    lines = lines(~cellfun(@isempty, lines));
    if numel(lines) < 2; error('apply:empty','%s has no data rows.', corrCsv); end
    hdr = strsplit(lines{1}, ',');
    ci  = @(nm) find(strcmp(hdr, nm), 1);
    cId = ci('record_id'); cQon = ci('qrs_onset_sample'); cQoff = ci('qrs_offset_sample');
    cTon = ci('t_onset_sample'); cToff = ci('t_offset_sample'); cSrc = ci('label_source');
    if any(cellfun(@isempty, {cId,cQon,cQoff,cTon,cToff}))
        error('apply:cols','%s is missing one of the boundary columns.', corrCsv);
    end

    cLead = ci('lead'); cQp = ci('q_peak_sample'); cRp = ci('r_peak_sample');
    cSp = ci('s_peak_sample'); cTp = ci('t_peak_sample');
    if isempty(cLead); cLead = cId; end

    copyfile(corrCsv, [corrCsv '.bak_' stamp]);
    moved = zeros(0,4); nrow = 0; touched = {}; warned = {};
    for k = 2:numel(lines)
        f = strsplit(lines{k}, ',', 'CollapseDelimiters', false);
        if numel(f) < numel(hdr); continue; end
        rec = f{cId};
        if ~isKey(B, rec); continue; end
        b   = B(rec);                       % [qOn qOff tOn tOff], 0-based samples
        old = [str2double(f{cQon}) str2double(f{cQoff}) str2double(f{cTon}) str2double(f{cToff})];
        new = old;
        if strcmp(mode,'all')
            new = b(1:4);              % b(5) is the trough, not a boundary
        else
            if isfinite(old(1)) && old(1) == 11;                new(1) = b(1); end
            if isfinite(old(3)) && isfinite(old(2)) && ...
               (old(3) <= old(2) || old(3) < b(5) - 2);         new(3) = b(3); end
        end
        if isequal(new, old); continue; end
        f{cQon}  = sprintf('%d', new(1));  f{cQoff} = sprintf('%d', new(2));
        f{cTon}  = sprintf('%d', new(3));  f{cToff} = sprintf('%d', new(4));
        if ~isempty(cSrc); f{cSrc} = 'manual_peaks_rule_bounds'; end
        % A peak that now falls outside its own window matters ONLY on the
        % reference lead, because that is the only lead exported. The other
        % eleven have their peaks recomputed from scratch by
        % propagate_to_all_leads. Checking here, where the row names its lead,
        % keeps the warning list to units you would actually have to look at.
        pk = {cQp,'Q',new(1),new(2); cRp,'R',new(1),new(2); ...
              cSp,'S',new(1),new(2); cTp,'T',new(3),new(4)};
        for pp = 1:size(pk,1)
            if isempty(pk{pp,1}); continue; end
            pv = str2double(f{pk{pp,1}});
            if ~isfinite(pv); continue; end
            if pv < pk{pp,3} || pv > pk{pp,4}
                warned{end+1} = sprintf('%s lead %s: %s peak at %d ms is outside the new [%d %d] window', ...
                    rec, f{cLead}, pk{pp,2}, pv, pk{pp,3}, pk{pp,4}); %#ok<AGROW>
            end
        end
        lines{k} = strjoin(f, ',');
        nrow = nrow + 1; touched{end+1} = rec; %#ok<AGROW>
        moved(end+1,:) = new - old; %#ok<AGROW>
    end
    tmp = [corrCsv '.tmp'];
    fid = fopen(tmp,'w');
    if fid < 0; error('apply:write','Cannot write %s', tmp); end
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
    movefile(tmp, corrCsv);

    % ---------------------------------------------------- rewrite the .mat
    nmat = 0;
    for i = 1:numel(ids)
        rec = ids{i};
        if ~isKey(B, rec); continue; end
        lf = fullfile(labelDir, [rec '_labels.mat']);
        if exist(lf,'file') ~= 2; continue; end
        b = B(rec);
        D = load(lf);
        if ~isfield(D,'L'); continue; end
        L = D.L; changed = false;
        for li = 1:numel(L)
            id = L(li).idx;
            if numel(id) < 8; continue; end
            nw = id;
            if strcmp(mode,'all')
                nw([1 5 6 8]) = b(1:4) + 1;      % tool holds 1-based indices
            else
                if isfinite(id(1)) && id(1) - 1 == 11;         nw(1) = b(1) + 1; end
                if isfinite(id(6)) && isfinite(id(5)) && ...
                   (id(6) <= id(5) || id(6) - 1 < b(5) - 2);   nw(6) = b(3) + 1; end
            end
            if isequal(nw, id); continue; end
            L(li).idx = nw;
            L(li).edited = true; L(li).reviewed = true;
            L(li).source = 'rule_bounds';
            changed = true;
        end
        if ~changed; continue; end
        if exist([lf '.bak_' stamp],'file') ~= 2; copyfile(lf, [lf '.bak_' stamp]); end
        meta = []; if isfield(D,'meta'); meta = D.meta; end %#ok<NASGU>
        if isempty(meta); save(lf,'L'); else; save(lf,'L','meta'); end
        nmat = nmat + 1;
    end

    % ---------------------------------------------------- report
    fprintf('rewrote %d CSV row(s) and %d per-record .mat file(s)\n', nrow, nmat);
    fprintf('backups: *.bak_%s\n\n', stamp);
    if ~isempty(moved)
        nm = {'QRS onset','QRS offset','T onset','T offset'};
        fprintf('how far each boundary moved (ms, + = later):\n');
        for j = 1:4
            v = moved(:,j); v = v(isfinite(v));
            if isempty(v); continue; end
            fprintf('   %-11s median %+5.0f   mean |move| %4.0f   worst %+5.0f\n', ...
                nm{j}, median(v), mean(abs(v)), v(find(abs(v)==max(abs(v)),1)));
        end
    end
    if ~isempty(warned)
        fprintf('\ncheck these by hand (%d) - reference leads only:\n', numel(warned));
        u = unique(warned);
        for i = 1:min(numel(u),20); fprintf('   %s\n', u{i}); end
    end
    fprintf('\nReopen manual_label_ecg to see the result, then run\n');
    fprintf('   propagate_to_all_leads\n   export_test_set_500hz\n\n');
end

% ======================================================================
function b = ruleBounds(M, LEADS, MAGLEADS)
%RULEBOUNDS  [qOn qOff tOn tOff jmin] as 0-BASED sample indices, from the
%   spatial magnitude across the eight independent leads at 5 per cent.
%   Identical arithmetic to the tool's own readout, including the QRS-offset
%   bound: the walk out of the QRS stops at the LOWEST point between the QRS
%   peak and the T peak, and the T peak is searched for only from 150 ms on,
%   because every QRS in this dataset is over by 135 ms and every T peak falls
%   after 210 ms. Without that bound the walk runs through the T on ischemia
%   records, bounding it at the first local minimum instead stops inside the
%   notch of a biphasic QRS.
    b = [];
    gi = find(ismember(LEADS, MAGLEADS));
    gi = gi(gi <= size(M,1));
    gi = gi(all(isfinite(M(gi,:)), 2));
    if numel(gi) < 4; return; end
    D = M(gi,:);
    D = D - median(D, 2, 'omitnan');
    D(~isfinite(D)) = 0;
    mag = sqrt(sum(D.^2, 1));
    nn  = numel(mag);
    if nn < 60; return; end
    fl  = qtile(mag, 0.10);

    [qpk, ipk] = max(mag);
    qamp = qpk - fl;
    if ~isfinite(qamp) || qamp <= 0; return; end
    thrQ = fl + 0.05*qamp;
    % Onset by scanning FORWARD from the start of the record for the first lift
    % off the floor. The back-walk stops
    % at the first sample below threshold going backwards, so on a complex with
    % a separated Q lobe it stops in the dip between Q and R and reports the
    % onset AFTER the Q - 22 ms late on AnteriorInfarction_013. Scanning forward
    % cannot do that, and these records genuinely start at isoelectric.
    k = find(mag > thrQ, 1, 'first');
    if isempty(k); k = ipk; end
    qOn = max(1, k - 1);
    jstop = nn;
    t0 = min(nn, max(ipk + 20, 151));
    if t0 < nn
        [~, r0] = max(mag(t0:nn));
        tp0 = t0 + r0 - 1;
        if tp0 > ipk + 2
            [~, r1] = min(mag(ipk:tp0));
            jstop = ipk + r1 - 1;
        end
    end
    i = ipk; while i < jstop && mag(i) > thrQ; i = i + 1; end
    qOff = i;

    lo = min(nn, qOff + 10);
    if lo >= nn - 2; return; end
    [pkv, rel] = max(mag(lo:nn));
    tp  = lo + rel - 1;
    amp = pkv - fl;
    if ~isfinite(amp) || amp <= 0; return; end
    i = tp; while i < nn && mag(i) > fl + 0.05*amp; i = i + 1; end
    tOff = i;

    seg = mag(qOff:tp);
    if isempty(seg); return; end
    [stLev, jrel] = min(seg);
    jmin = qOff + jrel - 1;
    amp2 = pkv - stLev;
    if ~isfinite(amp2) || amp2 <= 0; tOn = jmin; else
        thr2 = stLev + 0.05*amp2;
        i = jmin; while i < tp && mag(i) < thr2; i = i + 1; end
        tOn = i;
    end
    b = [qOn qOff tOn tOff jmin] - 1;     % to 0-based sample numbers
end

function v = qtile(x, p)
    s = sort(x(isfinite(x)));
    if isempty(s); v = NaN; return; end
    k = min(max(round(p*(numel(s)-1)) + 1, 1), numel(s));
    v = s(k);
end

% ======================================================================
function sig = loadSignals(matFile, LEADS)
%LOADSIGNALS  record_id -> struct(M, n, class) for every ECG table found.
    sig = containers.Map('KeyType','char','ValueType','any');
    raw = load(matFile);
    fn  = fieldnames(raw);
    for i = 1:numel(fn)
        v   = raw.(fn{i});
        cls = regexprep(char(fn{i}), 'ECGs?$', '');
        if iscell(v); items = v(:).'; else; items = {v}; end
        for k = 1:numel(items)
            T = items{k};
            if ~istable(T); continue; end
            vn = T.Properties.VariableNames;
            if any(strcmp(vn,'Time')); tc = double(T.Time);
            elseif any(strcmp(vn,'t')); tc = double(T.t);
            else; tc = (0:height(T)-1)'; end
            tc = tc(:).' - tc(1);
            M = nan(numel(LEADS), numel(tc));
            for j = 1:numel(LEADS)
                if any(strcmp(vn,LEADS{j})); c = double(T.(LEADS{j})); M(j,:) = c(:).'; end
            end
            sig(sprintf('%s_%03d',cls,k)) = struct('M',M,'n',numel(tc),'class',cls);
        end
    end
end
