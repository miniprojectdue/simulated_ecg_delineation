function repeat_review(mode, n)
%REPEAT_REVIEW  Measure how far the reviewer lands from themselves on a repeat.
%
%   repeat_review start      pick the units and blind them
%   repeat_review score      compare the second reading with the first
%   repeat_review restore    put the first reading back and forget the exercise
%
%   repeat_review('start', 30)   choose a different sample size, default 30
%
%   -- Why -----------------------------------------------------------------
%   Scored against the per-lead reference the network carries about 14 ms of T
%   offset scatter once a per-recording offset is removed. Two very different
%   things are inside that number. One is the network failing to track genuine
%   lead-to-lead variation. The other is the reviewer not landing in the same
%   place twice on a boundary that decays slowly and has no sharp end.
%
%   Only the first is a limitation of the model. The second is a property of the
%   reference and it puts a floor under what any delineator can score. They
%   cannot be separated by argument and they are cheap to separate by
%   measurement, which is what this does.
%
%   -- How -----------------------------------------------------------------
%   A fixed seed draws N units, spread evenly across the disease classes and at
%   most one per recording so the observations stay independent. Your placements
%   for those units are copied aside. The T boundaries are then reset to the
%   automatic delineator's values, which is an anchor that carries no
%   information about what you chose the first time, and the units are marked
%   unreviewed so the tool asks for them again.
%
%   Re-place them without looking at the saved file, then run the score step.
%   The number that comes out is the within-reviewer scatter, and it is the
%   floor. A model cannot be asked to agree with a reference more closely than
%   the reference agrees with itself.
%
%   Nothing is destroyed. The first reading lives in repeat_review_first_pass.csv
%   and restore puts it back, so the corpus is recoverable whatever you decide.

    here = fileparts(mfilename('fullpath'));
    labelDir = fullfile(here, 'labels');
    corrCsv  = fullfile(labelDir, 'smith2026_manual_corrections.csv');
    saveCsv  = fullfile(here, 'repeat_review_first_pass.csv');
    SEED     = 20260810;
    if nargin < 1 || isempty(mode); mode = 'start'; end
    if nargin < 2 || isempty(n);    n = 30;         end

    switch lower(strtrim(mode))
        case 'start';   doStart(labelDir, corrCsv, saveCsv, SEED, n);
        case 'score';   doScore(corrCsv, saveCsv);
        case 'restore'; doRestore(labelDir, corrCsv, saveCsv);
        otherwise
            error('repeat:mode','Mode is start, score or restore, not "%s".', mode);
    end
end

% ======================================================================
function doStart(labelDir, corrCsv, saveCsv, seed, n)
    if exist(saveCsv,'file') == 2
        error('repeat:already', ...
            ['%s already exists, so a repeat is already under way.\n' ...
             'Run repeat_review score to finish it, or repeat_review restore ' ...
             'to abandon it.'], saveCsv);
    end
    [hdr, rows] = readCsv(corrCsv);
    cRec = colOf(hdr,'record_id'); cLead = colOf(hdr,'lead');
    cCls = colOf(hdr,'disease_class');

    % one unit per recording, evenly across the classes, from a fixed seed
    byClass = containers.Map('KeyType','char','ValueType','any');
    recSeen = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(rows)
        p = split1(rows{i});
        rid = p{cRec};
        if ~isKey(recSeen, rid); recSeen(rid) = {}; end
        v = recSeen(rid); v{end+1} = i; recSeen(rid) = v; %#ok<AGROW>
        cls = p{cCls};
        if ~isKey(byClass, cls); byClass(cls) = {}; end
        w = byClass(cls); if ~ismember(rid, w); w{end+1} = rid; end %#ok<AGROW>
        byClass(cls) = w;
    end
    classes = keys(byClass);
    perClass = max(1, floor(n / numel(classes)));
    rng(seed);
    picked = {};
    for c = 1:numel(classes)
        recs = byClass(classes{c});
        ord  = randperm(numel(recs));
        take = min(perClass, numel(recs));
        for k = 1:take
            rid = recs{ord(k)};
            idxs = recSeen(rid);
            j = idxs{randi(numel(idxs))};
            picked{end+1} = j; %#ok<AGROW>
        end
    end
    fprintf('Drawn %d unit(s) with seed %d, at most one per recording.\n', numel(picked), seed);

    % keep the first reading before anything is touched
    fid = fopen(saveCsv,'w');
    fprintf(fid,'%s\n', strjoin(hdr,','));
    for a = 1:numel(picked); fprintf(fid,'%s\n', rows{picked{a}}); end
    fclose(fid);
    fprintf('First reading saved to %s\n\n', saveCsv);

    % blind the units in the label files
    cTon = colOf(hdr,'t_onset_sample'); cToff = colOf(hdr,'t_offset_sample');
    LE = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    fprintf('%-24s %-5s   first reading, T start and T end\n','record','lead');
    for a = 1:numel(picked)
        p = split1(rows{picked{a}});
        rid = p{cRec}; lead = p{cLead};
        fprintf('%-24s %-5s   %6s %6s\n', rid, lead, p{cTon}, p{cToff});
        lf = fullfile(labelDir, [rid '_labels.mat']);
        if exist(lf,'file') ~= 2
            warning('no label file for %s, skipped', rid); continue;
        end
        S = load(lf);
        li = find(strcmp(LE, lead), 1);
        if isempty(li) || li > numel(S.L); continue; end
        sd = nan(1, numel(S.L(li).idx));
        if isfield(S.L(li),'seed'); sd = S.L(li).seed; end
        S.L(li).idx(6) = pick(sd, 6, S.L(li).idx(6));
        S.L(li).idx(8) = pick(sd, 8, S.L(li).idx(8));
        S.L(li).tReviewed = false;
        L = S.L; %#ok<NASGU>
        if isfield(S,'meta'); meta = S.meta; save(lf,'L','meta'); else; save(lf,'L'); end %#ok<NASGU>
    end
    fprintf(['\nThose units now carry the automatic delineator''s T boundaries rather\n' ...
             'than yours. Open the tool, find each record and lead above, and place\n' ...
             'them again WITHOUT reading the column on the right.\n\n' ...
             '  manual_label_ecg(''SimulatedECGs_Smith2026_100.mat'', ''labels'', ''tlead'')\n\n' ...
             'Then run  repeat_review score\n']);
end

function v = pick(sd, k, fallback)
    v = fallback;
    if numel(sd) >= k && isfinite(sd(k)); v = sd(k); end
end

% ======================================================================
function doScore(corrCsv, saveCsv)
    if exist(saveCsv,'file') ~= 2
        error('repeat:none','No repeat under way. Run repeat_review start first.');
    end
    [h1, r1] = readCsv(saveCsv);
    [h2, r2] = readCsv(corrCsv);
    key = @(h,p) [p{colOf(h,'record_id')} '|' p{colOf(h,'lead')}];
    now = containers.Map('KeyType','char','ValueType','char');
    for i = 1:numel(r2); now(key(h2, split1(r2{i}))) = r2{i}; end

    dOn = []; dOff = []; pend = {};
    for i = 1:numel(r1)
        p1 = split1(r1{i}); k = key(h1,p1);
        if ~isKey(now,k); continue; end
        p2 = split1(now(k));
        a1 = getn(p1, colOf(h1,'t_onset_sample')); a2 = getn(p2, colOf(h2,'t_onset_sample'));
        b1 = getn(p1, colOf(h1,'t_offset_sample')); b2 = getn(p2, colOf(h2,'t_offset_sample'));
        if isfinite(a1) && isfinite(a2) && isfinite(b1) && isfinite(b2) ...
                && (a1 ~= a2 || b1 ~= b2)
            dOn(end+1)  = a2 - a1; %#ok<AGROW>
            dOff(end+1) = b2 - b1; %#ok<AGROW>
        else
            pend{end+1} = k; %#ok<AGROW>
        end
    end
    if ~isempty(pend)
        fprintf('%d unit(s) look unchanged, so they may not have been re-placed yet:\n', numel(pend));
        for a = 1:min(numel(pend),10); fprintf('   %s\n', pend{a}); end
        if numel(dOn) < 5
            fprintf('\nToo few re-placed units to say anything. Finish them and run this again.\n');
            return;
        end
        fprintf('\nScoring the %d that did change.\n\n', numel(dOn));
    end

    report('T start', dOn);
    report('T end',   dOff);
    fprintf(['\nThe T end figure is the one that matters. It is the floor under any\n' ...
             'delineator scored against this reference, since a model cannot agree\n' ...
             'with a reference more closely than the reference agrees with itself.\n' ...
             'Compare it with the network''s residual T offset scatter of 14.37 ms.\n']);
end

function report(name, d)
    if isempty(d); fprintf('%-8s no data\n', name); return; end
    a = sort(abs(d));
    q = @(p) a(max(1, min(numel(a), round(p*numel(a)))));
    fprintf('%-8s n %d   median |difference| %5.1f ms   IQR %.0f to %.0f   max %.0f\n', ...
            name, numel(d), median(a), q(0.25), q(0.75), a(end));
    fprintf('%-8s   signed mean %+.1f ms, sd %.1f\n', '', mean(d), std(d));
end

% ======================================================================
function doRestore(labelDir, corrCsv, saveCsv)
    if exist(saveCsv,'file') ~= 2
        error('repeat:none','Nothing to restore, %s does not exist.', saveCsv);
    end
    [h1, r1] = readCsv(saveCsv);
    [h2, r2] = readCsv(corrCsv);
    cRec = colOf(h1,'record_id'); cLead = colOf(h1,'lead');
    first = containers.Map('KeyType','char','ValueType','char');
    for i = 1:numel(r1)
        p = split1(r1{i}); first([p{cRec} '|' p{cLead}]) = r1{i};
    end
    out = cell(1,numel(r2)); nrep = 0;
    for i = 1:numel(r2)
        p = split1(r2{i}); k = [p{colOf(h2,'record_id')} '|' p{colOf(h2,'lead')}];
        if isKey(first,k); out{i} = first(k); nrep = nrep + 1; else; out{i} = r2{i}; end
    end
    stamp = datestr(now,'yyyymmdd_HHMMSS'); %#ok<TNOW1,DATST>
    copyfile(corrCsv, [corrCsv '.bak_repeat_' stamp]);
    fid = fopen(corrCsv,'w');
    fprintf(fid,'%s\n', strjoin(h2,','));
    for i = 1:numel(out); fprintf(fid,'%s\n', out{i}); end
    fclose(fid);

    LE = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    cTon = colOf(h1,'t_onset_sample'); cToff = colOf(h1,'t_offset_sample');
    for i = 1:numel(r1)
        p = split1(r1{i});
        lf = fullfile(labelDir, [p{cRec} '_labels.mat']);
        if exist(lf,'file') ~= 2; continue; end
        S = load(lf); li = find(strcmp(LE, p{cLead}), 1);
        if isempty(li) || li > numel(S.L); continue; end
        a = getn(p,cTon); b = getn(p,cToff);
        if isfinite(a); S.L(li).idx(6) = a + 1; end
        if isfinite(b); S.L(li).idx(8) = b + 1; end
        S.L(li).tReviewed = true;
        L = S.L; %#ok<NASGU>
        if isfield(S,'meta'); meta = S.meta; save(lf,'L','meta'); else; save(lf,'L'); end %#ok<NASGU>
    end
    delete(saveCsv);
    fprintf('Restored %d row(s) and their label files. The repeat is forgotten.\n', nrep);
end

% ======================================================================
function [hdr, rows] = readCsv(p)
    txt = fileread(p);
    lines = regexp(txt, '\r\n|\n|\r', 'split');
    lines = lines(~cellfun(@(x) isempty(strtrim(x)), lines));
    hdr = strtrim(strsplit(lines{1}, ',', 'CollapseDelimiters', false));
    rows = lines(2:end);
end

function c = colOf(hdr, name)
    c = find(strcmp(hdr, name), 1);
    if isempty(c); error('repeat:col','No %s column.', name); end
end

function p = split1(line)
    p = strsplit(line, ',', 'CollapseDelimiters', false);
end

function v = getn(p, c)
    v = NaN;
    if numel(p) < c; return; end
    t = strtrim(p{c});
    if ~isempty(t); v = str2double(t); end
end
