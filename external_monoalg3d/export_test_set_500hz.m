function export_test_set_500hz(matFile, corrCsv, outRoot, opts)
%EXPORT_TEST_SET_500HZ  Turn the manually reviewed Smith2026 records into a
%   500 Hz TEST set the ml_modelling loader can read.
%
%   export_test_set_500hz
%   export_test_set_500hz(MATFILE, CORRCSV, OUTROOT)
%   export_test_set_500hz(..., OPTS)
%
%   OPTS fields (all optional):
%       antiAlias        true   low-pass before decimating (see below)
%       fcHz             150    anti-alias cutoff, Hz
%       firOrder         64     FIR taps - 1, even so the kernel is symmetric
%       dropExcluded     true   skip leads flagged Exclude in the reviewer
%       requireReviewed  true   only export rows with reviewed = 1
%
%   Defaults, all relative to this file's folder:
%       MATFILE  SimulatedECGs_Smith2026.mat
%       CORRCSV  labels/smith2026_manual_corrections.csv
%       OUTROOT  test_export/
%
%   Writes ONLY inside OUTROOT. Nothing in the Delineation repo is read or
%   written by this script.
%
%   -- What comes out ----------------------------------------------------
%     test_export/signals/<record_id>_raw.csv
%         The 12 leads at 500 Hz, one lead per ROW, no header, which is the
%         layout ml_modelling/scripts/dataset.py reads with
%         np.loadtxt(..., delimiter=',') (it transposes if leads are columns).
%     test_export/smith2026_test_units.csv
%         One row per reviewed (record, lead), carrying the loader's required
%         columns and little else:
%           record_id lead beat_id split path_raw n_samples
%           win_start_sample win_end_sample
%           p_onset_sample p_peak_sample p_offset_sample
%           qrs_onset_sample q_peak_sample r_peak_sample s_peak_sample
%           qrs_offset_sample t_onset_sample t_peak_sample t_offset_sample
%         plus disease_class, fs_hz, the presence flags, and the *_ms columns
%         described below.
%
%   This is deliberately NOT the 43-column finetune schema. That table carries
%   ECGdeli provenance (conf_*, qc_status, rep_qc_status, unit_worst_status) that
%   this data has no equivalent of, because it never went through ECGdeli.
%   Inventing those columns would be fabricating provenance. The loader
%   hard-requires only the eight metadata columns above plus the eleven landmark
%   columns, so that is what is written.
%
%   -- Downsampling 1000 Hz -> 500 Hz ------------------------------------
%   500 Hz is NOT the same signal as 1000 Hz, and downsampling is not required
%   by the model in principle. It is done here for one specific reason: the
%   network was trained on 500 Hz MedalCare data, and feeding a test set at a
%   different sample rate would change the input time base the model has learned.
%   Mixing rates without resampling is the thing to avoid.
%
%   The conversion is the standard two-step one:
%     1. ANTI-ALIAS LOW-PASS at fcHz (default 150 Hz, the usual diagnostic ECG
%        bandwidth, comfortably under the 250 Hz Nyquist of the 500 Hz target).
%     2. Keep every second sample, so 500 Hz sample j is 1000 Hz sample 2j,
%        i.e. time 2j ms.
%
%   The filter is a windowed-sinc FIR with a Hamming window, applied with
%   conv(..., 'same'). The kernel is symmetric, so it has linear phase, and
%   'same' centring removes exactly its group delay: the result is ZERO phase
%   shift. That matters more here than anywhere else in the pipeline, because a
%   filter that moved the trace even one sample relative to the labels would
%   silently bias every boundary measurement. Signals are edge-padded with their
%   first and last value before filtering so the flat baseline either side of the
%   beat does not produce a start-up transient. No Signal Processing Toolbox
%   function is used, so this runs on a bare MATLAB install.
%
%   Set opts.antiAlias = false to decimate without filtering, which is useful
%   only for measuring how little difference it makes on this particular data.
%   The default is to filter: on the one record that can be inspected outside
%   MATLAB the energy above 150 Hz is about 0.0014 percent of the total, so the
%   filter removes almost nothing real, but that was one record used as a proxy
%   for 162, and a free safeguard should not be skipped on the strength of a
%   proxy measurement.
%
%   -- Label precision: two sets of columns -------------------------------
%   You review at 1 ms. The model can only resolve 2 ms. Rounding the labels to
%   500 Hz and then scoring against them would add up to 1 ms of avoidable error
%   to every boundary, on top of the model's own error, which is exactly the
%   quantity being measured. So both are written:
%
%     <landmark>_sample   0-based index at 500 Hz, = round(i/2). This is what the
%                         loader consumes to build its supervision windows.
%     <landmark>_ms       the ORIGINAL 1 ms time, un-rounded, from the review.
%                         Score against this to keep the label side exact.
%
%   The count of landmarks that had to be rounded is reported, so the size of
%   that effect is a known number rather than an assumption.
%
%   -- Things to check on the first run ----------------------------------
%   Reported at the end, because they are properties of the data rather than
%   faults for this script to fix:
%     * crop length. pretrain.yaml crops to 1280 samples; a record is about 501
%       at 500 Hz, so the whole record fits and win_* spans all of it.
%     * beat context. The training corpus cut beats out of 10 s recordings, so
%       each window had neighbouring beats either side. These records hold a
%       single beat with flat baseline around it. That is a real train/test
%       distribution difference and belongs in the write-up.
%     * class names. disease_class keeps this dataset's own labels (Healthy,
%       AnteriorInfarction, ...), which are NOT the pipeline's eight MedalCare
%       classes. Good for reporting, but do not feed it to anything expecting
%       that fixed vocabulary.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(matFile); matFile = fullfile(here,'SimulatedECGs_Smith2026.mat'); end
    if nargin < 2 || isempty(corrCsv); corrCsv = fullfile(here,'labels','smith2026_manual_corrections.csv'); end
    if nargin < 3 || isempty(outRoot); outRoot = fullfile(here,'test_export'); end
    if nargin < 4; opts = struct(); end
    if ~isfield(opts,'antiAlias');       opts.antiAlias       = true;  end
    if ~isfield(opts,'fcHz');            opts.fcHz            = 150;   end
    if ~isfield(opts,'firOrder');        opts.firOrder        = 64;    end
    if ~isfield(opts,'dropExcluded');    opts.dropExcluded    = true;  end
    if ~isfield(opts,'requireReviewed'); opts.requireReviewed = true;  end

    if exist(matFile,'file') ~= 2
        error('export:mat','Cannot find %s', matFile);
    end
    if exist(corrCsv,'file') ~= 2
        error('export:corr',['Cannot find %s\n' ...
            'Review some records in manual_label_ecg first - that is what writes it.'], corrCsv);
    end

    sigDir = fullfile(outRoot,'signals');
    if exist(sigDir,'dir') ~= 7; mkdir(sigDir); end

    LEADS = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};

    C = readtable(corrCsv,'TextType','string');
    need = {'record_id','lead','qrs_onset_sample','q_peak_sample','r_peak_sample', ...
            's_peak_sample','qrs_offset_sample','t_onset_sample','t_peak_sample', ...
            't_offset_sample'};
    miss = need(~ismember(need, C.Properties.VariableNames));
    if ~isempty(miss)
        error('export:cols','Corrections CSV is missing: %s', strjoin(miss,', '));
    end

    % Refuse to overwrite the single-convention reference export with a per-lead
    % one by accident.
    %
    % smith2026_test_units.csv is what build_external_units.py validates its
    % 1,944-row table against, and that check is the only thing standing between
    % a mistake in the propagation and a silently wrong test set. Once the
    % per-lead T pass has run, the corrections CSV holds twelve rows per record
    % rather than one, and re-running this script would rewrite the reference
    % from those rows, so the check would then be comparing the new table against
    % itself and would pass no matter what. Set opts.allowPerLeadT to override,
    % having first copied the existing export somewhere safe.
    if ~isfield(opts,'allowPerLeadT'); opts.allowPerLeadT = false; end
    if ~opts.allowPerLeadT && ismember('label_source', C.Properties.VariableNames)
        isTL = arrayfun(@(i) strcmpi(strtrim(char(string(C.label_source(i)))), ...
                        'manual_tlead_perlead_T'), (1:height(C)).');
        if any(isTL)
            error('export:perleadT', ...
                ['%s holds %d per-lead T rows, written by the per-lead review.\n' ...
                 'Re-running this script would rewrite the single-convention\n' ...
                 'reference export that build_external_units.py checks against.\n' ...
                 'Run propagate_to_all_leads then build_external_units.py instead.\n' ...
                 'To override, copy test_export/smith2026_test_units.csv aside and\n' ...
                 'pass opts.allowPerLeadT = true.'], corrCsv, sum(isTL));
        end
    end

    raw  = load(matFile);
    recs = expandRecords(raw, LEADS);
    if isempty(recs)
        error('export:norecs','No ECG tables found in %s', matFile);
    end
    byName = containers.Map('KeyType','char','ValueType','any');
    for i = 1:numel(recs); byName(recs(i).name) = recs(i); end

    LAND = {'p_onset_sample','p_peak_sample','p_offset_sample', ...
            'qrs_onset_sample','q_peak_sample','r_peak_sample','s_peak_sample', ...
            'qrs_offset_sample','t_onset_sample','t_peak_sample','t_offset_sample'};
    MSCOL = strrep(LAND, '_sample', '_ms');
    cols = [{'record_id','disease_class','lead','beat_id','split','fs_hz', ...
             'n_samples','path_raw','win_start_sample','win_end_sample'}, LAND, MSCOL, ...
            {'p_present','qrs_present','q_present','r_present','s_present','t_present'}];

    rows = {};
    cache   = containers.Map('KeyType','char','ValueType','any');   % rid -> n500
    nRounded = 0; nLand = 0; nExcluded = 0; nMissingSig = 0;
    recSeen = containers.Map('KeyType','char','ValueType','logical');

    for r = 1:height(C)
        rid  = char(C.record_id(r));
        lead = char(C.lead(r));

        if opts.requireReviewed && ismember('reviewed', C.Properties.VariableNames)
            if getNum(C.reviewed(r)) ~= 1; continue; end
        end
        if opts.dropExcluded && ismember('exclude', C.Properties.VariableNames)
            if getNum(C.exclude(r)) == 1; nExcluded = nExcluded + 1; continue; end
        end
        if ~isKey(byName, rid); nMissingSig = nMissingSig + 1; continue; end

        rec     = byName(rid);
        relName = [rid '_raw.csv'];
        sigPath = fullfile(sigDir, relName);
        if ~isKey(cache, rid)
            M500 = downsampleHalf(rec.M, 1000, opts);
            writeMatrixCsv(sigPath, M500);
            cache(rid) = size(M500,2);
        end
        n500 = cache(rid);

        vals = cell(1, numel(cols));
        for c = 1:numel(cols)
            nm = cols{c};
            switch nm
                case 'record_id';        vals{c} = rid;
                case 'disease_class';    vals{c} = rec.class;
                case 'lead';             vals{c} = lead;
                case 'beat_id';          vals{c} = '1';
                case 'split';            vals{c} = 'test';
                case 'fs_hz';            vals{c} = '500';
                case 'n_samples';        vals{c} = num2str(n500);
                case 'path_raw';         vals{c} = strrep(sigPath, ',', '_');
                case 'win_start_sample'; vals{c} = '0';
                case 'win_end_sample';   vals{c} = num2str(n500 - 1);
                otherwise
                    if ismember(nm, LAND)
                        v = colNum(C, nm, r);
                        if isfinite(v)
                            nLand = nLand + 1;
                            if mod(round(v),2) ~= 0; nRounded = nRounded + 1; end
                            vals{c} = num2str(round(v/2));
                        else
                            vals{c} = '';
                        end
                    elseif ismember(nm, MSCOL)
                        % exact 1 ms ground truth, from the 1000 Hz review
                        v = colNum(C, strrep(nm,'_ms','_sample'), r);
                        if isfinite(v); vals{c} = num2str(v); else; vals{c} = ''; end
                    else
                        v = colNum(C, nm, r);
                        if isfinite(v); vals{c} = num2str(v); else; vals{c} = '0'; end
                    end
            end
        end
        rows{end+1} = strjoin(vals, ','); %#ok<AGROW>
        recSeen(rid) = true;
    end

    if isempty(rows)
        error('export:empty', ...
            ['No reviewed rows found in %s.\n' ...
             'Review at least one record in manual_label_ecg first.'], corrCsv);
    end

    unitsCsv = fullfile(outRoot,'smith2026_test_units.csv');
    fid = fopen(unitsCsv,'w');
    if fid < 0; error('export:write','Cannot write %s', unitsCsv); end
    fprintf(fid,'%s\n', strjoin(cols,','));
    for a = 1:numel(rows); fprintf(fid,'%s\n', rows{a}); end
    fclose(fid);

    fprintf('\nWrote %s\n', unitsCsv);
    fprintf('  units (record,lead) rows : %d\n', numel(rows));
    fprintf('  records                  : %d\n', numel(keys(recSeen)));
    fprintf('  signal CSVs at 500 Hz    : %d  -> %s\n', numel(keys(cache)), sigDir);
    if opts.antiAlias
        fprintf('  anti-alias               : zero-phase FIR, fc = %g Hz, %d taps\n', ...
            opts.fcHz, opts.firOrder + 1);
    else
        fprintf('  anti-alias               : OFF (plain decimation)\n');
    end
    fprintf('  landmarks written        : %d\n', nLand);
    if nLand > 0
        fprintf('  rounded to 500 Hz grid   : %d  (%.1f%%) - score on the *_ms columns to avoid this\n', ...
            nRounded, 100*nRounded/nLand);
    end
    if nExcluded   > 0; fprintf('  skipped, lead excluded   : %d\n', nExcluded); end
    if nMissingSig > 0; fprintf('  skipped, no signal found : %d\n', nMissingSig); end
    fprintf('\nP columns are empty by design: this data has no P wave.\n');
    fprintf('path_raw is absolute, so the loader resolves it with no change to Delineation.\n\n');
end

% ======================================================================
function M2 = downsampleHalf(M, fs, opts)
%DOWNSAMPLEHALF  Anti-alias low-pass then keep every second sample.
%   Returns leads x ceil(N/2). 500 Hz sample j corresponds to 1000 Hz sample 2j,
%   so a landmark at 1000 Hz index i maps to round(i/2) with no time shift.
    if opts.antiAlias
        M = lowpassZeroPhase(M, fs, opts.fcHz, opts.firOrder);
    end
    M2 = M(:, 1:2:end);
end

function Y = lowpassZeroPhase(X, fs, fc, ord)
%LOWPASSZEROPHASE  Windowed-sinc FIR low-pass with EXACTLY zero phase shift.
%   A symmetric FIR kernel has linear phase, i.e. a pure delay of ord/2 samples.
%   conv(..., 'same') returns the centred output and so removes that delay
%   exactly, leaving no net shift. Zero phase is essential here: a filter that
%   moved the trace relative to the reviewed labels would bias every boundary.
%
%   Rolled by hand rather than using fir1/filtfilt so no Signal Processing
%   Toolbox licence is needed.
    if mod(ord,2) ~= 0; ord = ord + 1; end        % even order -> symmetric kernel
    m = (0:ord) - ord/2;
    ft = fc / fs;                                  % normalised cutoff (cycles/sample)
    h = 2*ft * sincLocal(2*ft*m);                  % ideal low-pass impulse response
    w = 0.54 - 0.46*cos(2*pi*(0:ord)/ord);         % Hamming window
    h = h .* w;
    h = h / sum(h);                                % unity DC gain
    pad = ord;
    Y = zeros(size(X));
    for i = 1:size(X,1)
        v = X(i,:);
        if all(~isfinite(v)); Y(i,:) = v; continue; end
        v(~isfinite(v)) = 0;
        vp = [repmat(v(1),1,pad), v, repmat(v(end),1,pad)];   % hold the edges
        yp = conv(vp, h, 'same');
        Y(i,:) = yp(pad+1 : pad+numel(v));
    end
end

function y = sincLocal(x)
%SINCLOCAL  sin(pi x)/(pi x) with the removable singularity handled.
    y = ones(size(x));
    nz = x ~= 0;
    y(nz) = sin(pi*x(nz)) ./ (pi*x(nz));
end

function writeMatrixCsv(path, M)
%WRITEMATRIXCSV  One lead per row, comma separated, NO header, matching the
%   layout the ml_modelling signal loader expects.
    fid = fopen(path,'w');
    if fid < 0; error('export:sig','Cannot write %s', path); end
    fmt = [repmat('%.8g,',1,size(M,2)-1) '%.8g\n'];
    for i = 1:size(M,1)
        fprintf(fid, fmt, M(i,:));
    end
    fclose(fid);
end

function recs = expandRecords(raw, LEADS)
%EXPANDRECORDS  One entry per ECG table in the .mat, named <Class>_NNN, matching
%   the naming manual_label_ecg uses so the corrections CSV joins on record_id.
    recs = struct('name',{},'class',{},'M',{});
    fn = fieldnames(raw);
    for i = 1:numel(fn)
        v = raw.(fn{i});
        cls = regexprep(char(fn{i}), 'ECGs?$', '');
        items = {};
        if iscell(v)
            items = v(:).';
        elseif isstruct(v) && numel(v) > 1
            items = arrayfun(@(e) e, v(:).', 'UniformOutput', false);
        elseif istable(v) || (isstruct(v) && isscalar(v))
            items = {v};
        end
        for k = 1:numel(items)
            M = leadMatrix(items{k}, LEADS);
            if isempty(M); continue; end
            recs(end+1) = struct('name',sprintf('%s_%03d',cls,k), ...
                                 'class',cls,'M',M); %#ok<AGROW>
        end
    end
end

function M = leadMatrix(x, LEADS)
%LEADMATRIX  12 x N matrix of leads from one ECG table or struct, [] if not one.
    M = [];
    if istable(x)
        vn = x.Properties.VariableNames;
    elseif isstruct(x) && isscalar(x)
        vn = fieldnames(x).';
    else
        return
    end
    have = LEADS(ismember(LEADS, vn));
    if numel(have) < 6; return; end
    first = double(getField(x, have{1}));
    N = numel(first);
    M = nan(numel(LEADS), N);
    for j = 1:numel(LEADS)
        if ismember(LEADS{j}, vn)
            col = double(getField(x, LEADS{j}));
            col = col(:).';
            if numel(col) == N; M(j,:) = col; end
        end
    end
end

function v = getField(x, nm)
%GETFIELD  Column nm from a table or struct, uniformly.
    v = x.(nm);
end

function v = colNum(T, nm, r)
%COLNUM  Numeric value of column nm at row r, NaN when absent or blank.
    v = NaN;
    if ismember(nm, T.Properties.VariableNames)
        v = getNum(T.(nm)(r));
    end
end

function v = getNum(x)
%GETNUM  Scalar double from a table cell, NaN when blank.
    if isnumeric(x); v = double(x); if isempty(v); v = NaN; end; return; end
    s = strtrim(string(x));
    if s == "" || ismissing(s) || s == "None"; v = NaN; else; v = str2double(s); end
end
