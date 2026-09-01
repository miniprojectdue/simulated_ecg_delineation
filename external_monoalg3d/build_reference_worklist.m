function build_reference_worklist(matFile, outCsv)

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(matFile); matFile = fullfile(here,'SimulatedECGs_Smith2026.mat'); end
    if nargin < 2 || isempty(outCsv);  outCsv  = fullfile(here,'smith2026_worklist.csv'); end
    if exist(matFile,'file') ~= 2; error('build:mat','Cannot find %s', matFile); end

    LEADS    = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    MAGLEADS = {'I','II','V1','V2','V3','V4','V5','V6'};

    raw = load(matFile);
    fn  = fieldnames(raw);
    recs = struct('name',{},'class',{},'lead',{},'mad',{});

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
            if all(all(~isfinite(M))); continue; end

            % 1. reconciled reference from the spatial magnitude
            gi = find(ismember(LEADS, MAGLEADS));
            gi = gi(all(isfinite(M(gi,:)),2));
            if numel(gi) < 4; continue; end
            Dm  = M(gi,:) - median(M(gi,:), 2, 'omitnan');
            Dm(~isfinite(Dm)) = 0;
            mag = sqrt(sum(Dm.^2, 1));
            G   = boundsFrom(mag, tc);
            if isempty(G); continue; end

            % 2-3. score each lead against it
            bestLead = ''; bestMad = Inf;
            for j = 1:numel(LEADS)
                if ~any(isfinite(M(j,:))); continue; end
                curve = abs(M(j,:) - median(M(j,:), 'omitnan'));
                B = boundsFrom(curve, tc);
                if isempty(B); continue; end
                md = mean(abs(B - G));
                if md < bestMad; bestMad = md; bestLead = LEADS{j}; end
            end
            if isempty(bestLead); bestLead = 'II'; bestMad = NaN; end

            recs(end+1) = struct('name',sprintf('%s_%03d',cls,k), ...
                'class',cls,'lead',bestLead,'mad',bestMad); %#ok<AGROW>
        end
    end

    if isempty(recs); error('build:none','No records found in %s', matFile); end

    % 4. interleave classes so a partial run stays balanced
    classes = unique({recs.class});
    byc = cell(1,numel(classes));
    for c = 1:numel(classes); byc{c} = find(strcmp({recs.class}, classes{c})); end
    order = []; p = ones(1,numel(classes));
    while true
        moved = false;
        for c = 1:numel(classes)
            if p(c) <= numel(byc{c})
                order(end+1) = byc{c}(p(c)); p(c) = p(c)+1; moved = true; %#ok<AGROW>
            end
        end
        if ~moved; break; end
    end

    fid = fopen(outCsv,'w');
    if fid < 0; error('build:write','Cannot write %s', outCsv); end
    fprintf(fid,'priority,record_id,disease_class,lead,ref_lead_mad_ms\n');
    for a = 1:numel(order)
        r = recs(order(a));
        fprintf(fid,'%d,%s,%s,%s,%.1f\n', a, r.name, r.class, r.lead, r.mad);
    end
    fclose(fid);

    fprintf('\nWrote %s  (%d units, one lead per record)\n', outCsv, numel(recs));
    fprintf('\nchosen lead distribution:\n');
    for j = 1:numel(LEADS)
        n = sum(strcmp({recs.lead}, LEADS{j}));
        if n > 0; fprintf('   %-4s %3d  %s\n', LEADS{j}, n, repmat('#',1,n)); end
    end
    m = [recs.mad]; m = m(isfinite(m));
    if ~isempty(m)
        ms = sort(m);
        fprintf('\nref_lead_mad_ms: median %.1f   p90 %.1f   max %.1f\n', ...
            ms(max(1,round(0.50*end))), ms(max(1,round(0.90*end))), ms(end));
        fprintf('(how far the chosen lead sits from the reconciled label - a large\n');
        fprintf(' value means NO lead shows this record''s boundaries well, so read\n');
        fprintf(' it off the orange curve and treat the foreground trace as advisory.)\n');
    end
end

% ======================================================================
function B = boundsFrom(curve, t)
%BOUNDSFROM  [QRSon QRSoff Ton Toff] in ms from a non-negative activity curve,
%   each taken at 5 percent of the relevant peak above the curve's noise floor.
%   Returns [] when no T hump is resolvable.
    B = [];
    curve = curve(:).';
    n = numel(curve);
    if n < 50; return; end
    sc = sort(curve(isfinite(curve)));
    if isempty(sc); return; end
    fl = sc(min(max(round(0.10*(numel(sc)-1))+1,1),numel(sc)));
    [pk, ip] = max(curve);
    if ~isfinite(pk) || pk <= fl; return; end
    thr = fl + 0.05*(pk - fl);
    i = ip; while i > 1 && curve(i) > thr; i = i - 1; end
    qon = t(i);
    i = ip; while i < n && curve(i) > thr; i = i + 1; end
    qoff = t(i);
    j0 = i + 10;
    if j0 >= n-5; return; end
    [tpv, rel] = max(curve(j0:n));
    tp  = j0 + rel - 1;
    amp = tpv - fl;
    if ~isfinite(amp) || amp <= 0; return; end
    thr2 = fl + 0.05*amp;
    i = tp; while i > j0 && curve(i) > thr2; i = i - 1; end
    ton = t(i);
    i = tp; while i < n && curve(i) > thr2; i = i + 1; end
    toff = t(i);
    B = [qon qoff ton toff];
end
