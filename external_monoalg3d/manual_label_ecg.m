function manual_label_ecg(inputPath, outDir, mode)
%MANUAL_LABEL_ECG  Interactive manual fiducial-point labelling for the
%   internal (simulated) ECG dataset, carrying the multi-lead review aids
%   that the MedalCare gold-standard tool (gold_label_ecg.m) uses.
%
%   manual_label_ecg                 % pick a .mat file or folder via dialog
%   manual_label_ecg(FILE)           % label a single ecg_table-style .mat
%   manual_label_ecg(FOLDER)         % label every *.mat table in a folder
%   manual_label_ecg(FILE,  OUTDIR)  % choose where the label files are saved
%   manual_label_ecg(FOLDER,OUTDIR)
%   manual_label_ecg(FILE, OUTDIR, 'tlead')   % per-lead T review, see below
%
%   -- Per-lead T review mode ('tlead') ----------------------------------
%   The default review places one boundary set per record, read off the shared
%   spatial magnitude curve, and propagates it to all twelve leads. That is the
%   right convention for a boundary defined on the beat, and it is what
%   apply_boundary_rule and propagate_to_all_leads implement.
%
%   It is NOT the convention the training corpus uses for the T wave. There the
%   T boundaries are per-lead human placements read off each lead's own trace at
%   the return to the isoelectric baseline, and the two conventions disagree by a
%   fixed displacement that dominates the external T-offset error. Passing
%   'tlead' switches the tool into a mode that closes that gap without redoing
%   the QRS work:
%
%     * all twelve leads of every record are in scope, not the worklist's one
%     * QRS onset and QRS offset are LOADED FROM THE CORRECTIONS CSV, identical
%       in every lead, exactly as the current test set carries them. They stay
%       editable, and an edit applies to all twelve leads at once
%     * T start and T end are yours to place, per lead, off that lead's own
%       black trace. Every lead opens holding the labels the earlier pass left
%       on it. Nothing is seeded and nothing is computed for you
%     * T peak stays editable since it is already a per-lead quantity
%     * the readout gains a per-lead reference figure at TFRAC of that lead's own
%       T amplitude above its own baseline, alongside the shared 'mag 5%%'
%       figures, so the two conventions can be read side by side. Both are
%       printed and neither is applied to any label
%     * 'b' copies QRS ONLY, so it can never overwrite per-lead T work
%     * all twelve leads are written for every record you leave
%
%   The criterion behind the per-lead suggestion is a decision, not a
%   measurement. It has to be the criterion the training corpus used, otherwise
%   the relabelling fixes the scope difference and leaves the convention
%   difference in place. Read SMITH2026_TLEAD_ADDENDUM.md before starting.
%
%   Each ECG record is a MATLAB table with a time column ('Time' or 't') and
%   12 lead columns named exactly:
%       I II III aVR aVL aVF V1 V2 V3 V4 V5 V6
%   i.e. the same layout as ecg_table.mat that ships in this repo. The sample
%   interval is read from the time column rather than assumed, so a 1 ms record
%   and a 2 ms record both display and save correctly.
%
%   An input .mat may hold that record in three ways, all handled automatically:
%     * a single table / single ECG struct  (ecg_table.mat, all_1_table.mat)
%     * a folder of such single-record files (labelled one after another)
%     * a MULTI-record file whose variables are cell arrays of ECG tables, one
%       cell array per disease class - this is the SimulatedECGs_Smith2026.mat
%       layout (HealthyECGs, AnteriorInfarctionECGs, AnteriorIschemiaECGs,
%       InferiorInfarctionECGs, InferiorIschemiaECGs). Every table becomes its
%       own record, named <Class>_NNN and tagged with its class. So point the
%       tool at SimulatedECGs_Smith2026.mat directly to review all of it; it is
%       not limited to ecg_table.mat / all_1_table.mat.
%
%   -- Choosing a record -------------------------------------------------
%   The left column is the record browser, on the same pattern as the Gold
%   tool's unit list: a class popup at the top narrows the list to one disease
%   class, and clicking a row opens that record. A record that already has a
%   label file on disk is marked with *, so you can see how far through a class
%   you are. Ctrl+n / Ctrl+p step within the FILTERED list, so with a class
%   selected they walk that class only. The lead picker sits immediately to its
%   right and the plot title shows record name, class, lead and rec i/N.
%
%   -- Why this version exists -------------------------------------------
%   The first version of this tool drew one lead on its own, with no zoom
%   and no cross-lead reference. That made onsets and offsets systematically
%   LATE and EARLY respectively. A lead whose axis sits near perpendicular
%   to the initial depolarisation vector registers almost nothing for the
%   first several ms of the QRS, so the eye places the onset after the wave
%   has already begun elsewhere. The error only ever runs one way, so it
%   does not average out across leads. Three aids fix that, all ported from
%   the gold-standard tool:
%
%     1. SPATIAL MAGNITUDE STRIP (orange, drawn under the trace). The root
%        sum of squares across the eight INDEPENDENT leads I, II and V1..V6,
%        each baseline-removed first. III, aVR, aVL and aVF are exact linear
%        combinations of I and II, so including them would triple-count the
%        frontal plane. Squaring removes polarity and summing over eight
%        axes removes perpendicularity, so the curve lifts off its floor the
%        instant ANY lead starts moving. The dotted orange line is the noise
%        floor, taken as the 10th centile of the curve.
%     2. TWELVE-LEAD OVERLAY (grey). Every other lead amplitude-normalised
%        onto the lead being edited, so a landmark that looks ambiguous in
%        one lead can be read against the rest.
%     3. ZOOM AND PAN. Scroll wheel over the trace or the +/- keys to zoom,
%        , and . to pan, and f to return to the full-beat fit.
%
%   -- How to place a landmark -------------------------------------------
%   ONSETS and OFFSETS (QRS start, QRS end, T start, T end) come from the
%   ORANGE magnitude curve. The onset is the first sustained lift off the
%   floor, the offset is the last return to it.
%   PEAKS (Q, R, S, T peak) come from the BLACK single-lead trace, because a
%   peak is a property of that lead's own projection and the magnitude curve
%   has had its polarity removed.
%
%   -- Landmarks ---------------------------------------------------------
%       QRS start | Q | R | S | QRS end | T start | T peak | T end
%
%   There is no P wave: the simulated data does not contain one, per
%   README.md. Q, R and S are named on the Gold convention, against the
%   QRS-onset voltage as the isoelectric reference and a threshold of five
%   per cent of the largest deflection inside the complex. A landmark that
%   is genuinely absent is stored blank rather than guessed, so a QS complex
%   saves a Q with no R and no S.
%
%   -- Adjusting a point -------------------------------------------------
%     * DRAG any marker along the trace (it snaps to the nearest sample)
%     * or press 1..8 to SELECT a landmark, then:
%         LEFT / RIGHT arrow  nudge it by one sample (hold Shift = 5)
%         CLICK on the trace  jump the selected landmark to that x
%     * DELETE / BACKSPACE clears the selected landmark (marks it absent)
%     * "T absent" clears (or restores) the whole T wave. "Q absent",
%       "R absent" and "S absent" do the same for one QRS peak each: press
%       once to mark that wave absent, press again to restore it from a fresh
%       seed. Use them when the complex has no such wave, for example no R in
%       a QS complex, no Q in an rS complex, or neither Q nor S in a
%       monophasic R.
%
%   -- Keyboard ----------------------------------------------------------
%     (or click a landmark in the right-hand list to select it)
%     1..8     select QRS start / Q / R / S / QRS end / T start / T pk / T end
%     < >      (left/right arrows) nudge the selected landmark
%     Del/Bksp clear the selected landmark (absent)
%     o        toggle the 12-lead overlay
%     m        toggle the spatial magnitude strip
%     + / -    zoom in / out         , / .   pan left / right
%     f or 0   restore the full-beat fit
%     n / p    next / previous lead
%     g        show the tangent-method T-offset construction: the steepest
%              limb after the T peak extended to the isoelectric baseline,
%              with the crossing marked and its distance from the current
%              T end printed. Shift+G moves T end onto that crossing.
%              CAUTION on THIS dataset. These simulated T waves end with a
%              long slow tail: the steep limb finishes near 340 ms but real
%              signal continues to about 500 ms, at which point the amplitude
%              is still an order of magnitude above baseline. The tangent
%              extrapolates the steep limb and so cuts roughly 150 ms early
%              here. Prefer the 'mag 5%% Toff' figure in the readout, which
%              tracks the tail. The tangent remains the right construction for
%              a T wave that ends cleanly, which is not this one.
%     b        copy this lead's four BOUNDARIES (QRS onset, QRS offset, T onset,
%              T offset) onto all twelve leads, leaving each lead's own peaks
%              untouched. A boundary belongs to the beat rather than to one
%              lead, and the training corpus stores exactly one value per
%              boundary per record, so decide them once on the clearest lead
%              and press b. Disabled in tlead mode, where QRS is already shared
%              and locked and T is per lead by design.
%     Shift+B  copy the QRS boundaries only, leaving both T boundaries alone.
%     u        (tlead mode) hold the QRS boundaries still for THIS record, so a
%              stray drag near the J point cannot move them. Everything is
%              editable by default and u is only a guard. A QRS edit always
%              applies to all twelve leads whether or not the guard is on.
%     r        re-seed THIS lead from the automatic delineator
%     t        toggle the whole T wave present / absent
%     i        toggle the "inverted T" flag for this lead
%     x        toggle the "exclude this lead" flag
%     s        save all labels for the current record
%     Ctrl+n / Ctrl+p   next / previous record (when a folder was given)
%
%   -- Output ------------------------------------------------------------
%   Everything lands in OUTDIR (default <input folder>/labels). Nothing outside
%   this folder is ever written.
%
%     smith2026_manual_corrections.csv   ONE combined table for all records, in
%         the same column layout the Gold reviewer tool emits, so the existing
%         Delineation scripts can read it by column name without being taught a
%         new schema:
%           record_id disease_class lead beat_id fs_hz n_samples
%           p_onset_sample p_peak_sample p_offset_sample
%           qrs_onset_sample q_peak_sample r_peak_sample s_peak_sample
%           qrs_offset_sample t_onset_sample t_peak_sample t_offset_sample
%           p_present qrs_present q_present r_present s_present t_present
%           flags also_delineator priority label_source reviewed edited_at
%           invertedT exclude
%         One row per (record, lead). Sample indices are 0-based, matching the
%         pipeline convention. A blank landmark means reviewed-as-absent and is
%         paired with its *_present = 0, never a guessed number. The three P
%         columns are always blank with p_present = 0, because this simulated
%         data contains no P wave. invertedT and exclude are appended beyond the
%         shared 29 columns since the Gold layout has no slot for them; a
%         name-based reader simply ignores them.
%         Saves MERGE into this file, so reviewing one record never disturbs
%         another's rows, and the write goes via a temp file so an interrupted
%         save cannot truncate it.
%
%     <record>_labels.mat   per-record working state, what the tool resumes from.
%
%   Nothing is written until you actually do something. Opening a record and
%   looking at its automatic seed writes no file and leaves it unmarked, so *
%   means "reviewed by you" and not "opened once".
%
%   There is nothing to press. Two things write to disk:
%
%     * an EDIT - moving, clearing or restoring a landmark, or toggling a flag -
%       is written the moment you make it. A drag writes once when you release
%       the mouse rather than on every mouse-move, so a gesture costs one write
%       and not fifty.
%     * LEAVING a record writes all twelve of its leads. Leads you corrected
%       carry your values; leads you left alone carry the automatic seed, which
%       is you accepting it as correct.
%
%   Arriving at a record writes nothing, only leaving it does, so a record you
%   never open stays out of the export and the file can never contain delineator
%   output you never looked at.
%
%   NOTE, on feeding the ML pipeline. These rows are deliberately NOT presented as
%   drop-in rows for the finetune table in Delineation/. That table is 500 Hz
%   MedalCare data with a real P wave, a path_raw pointing at a per-record CSV,
%   and win_start_sample / win_end_sample crop columns. This data is 1000 Hz, has
%   no P wave, lives inside a .mat, and uses its own class names. Converting
%   between the two is a separate, deliberate step.
%
%   Requires delineate_ecg_v3.m and get_window.m on the MATLAB path (both
%   ship in this same folder). Tested against R2019b+; needs the classic
%   figure GUI (not MATLAB Online's -nodisplay mode).
%
%   Hannah Smith & Maxx Holmes delineation code is used for the automatic
%   pre-seed. The review aids are adapted from gold_label_ecg.m, which is
%   itself a patched copy of manual_labelling/tool/medalcare_label_ecg.m.

% ------------------------------------------------------------------ setup
LEADS = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};

% Landmarks in TEMPORAL order. Q, S and the whole T group may be blank.
FIDS   = {'QRS start','Q','R','S','QRS end','T start','T peak','T end'};
FIDKEY = {'QRS_start','Q_peak','R_peak','S_peak','QRS_end', ...
          'T_start','T_peak','T_end'};
FLABEL = {'QRSon','Q','R','S','QRSoff','Ton','Tpk','Toff'};
FIDCOL = [0.85 0.10 0.10;   % QRS start  red
          0.00 0.45 0.45;   % Q          teal
          0.90 0.10 0.55;   % R          magenta
          0.55 0.35 0.10;   % S          brown
          0.90 0.45 0.00;   % QRS end    orange
          0.10 0.55 0.85;   % T start    blue
          0.20 0.65 0.20;   % T peak     green
          0.55 0.20 0.70];  % T end      purple

QGRP = 1:5;                 % QRS start .. QRS end
TGRP = 6:8;                 % T start .. T end
CORE = [1 3 5 6 7 8];       % the landmarks the auto-seed always supplies

% The eight INDEPENDENT leads. III, aVR, aVL and aVF are exact linear
% combinations of I and II, so adding them would weight the frontal plane
% three times over without adding one new spatial axis.
MAGLEADS = {'I','II','V1','V2','V3','V4','V5','V6'};

WIDTH      = 3;    % sliding-window width passed to delineate_ecg_v3
ACTIVATION = 67;   % ms; activation-time argument for delineate_ecg_v3

% Per-lead T criterion, used only in 'tlead' mode. A T boundary is placed where
% that lead's own trace falls within TFRAC of the T amplitude of that lead's own
% isoelectric baseline.
%
% The value is MEASURED, not chosen. Applying this family of criteria to 270
% human-placed T offsets in the MedalCare-XL reference, against each lead's own
% pre-QRS baseline and inside the reviewer's own beat window, gives the median
% signed difference below. Positive means the human placed the boundary later
% than the criterion.
%
%     fraction   T offset      T onset
%     0.02        +4 ms        -16 ms
%     0.05       +10 to +12    -37
%     0.10       +16 to +18    -64
%     0.20       +24 to +26    -95
%     tangent    +19 to +21
%
% Two per cent reproduces the training reference to within one 500 Hz sample on
% the T offset and is the closest of the family on the T onset, where the scatter
% is far wider and the boundary is genuinely fuzzy. The tangent, which the
% MedalCare protocol names, lands about 20 ms early on its own reference.
%
% An earlier version of this constant was three standard deviations of the
% record's tail noise. That was wrong for this corpus. These beats are simulated
% with almost no numerical noise in the tail, so the threshold collapsed toward
% zero and the walk ran to 779 ms on AnteriorInfarction_001 where every member of
% the threshold family sits below 522 ms. A criterion referenced to the wave's
% own amplitude does not have that failure mode.
%
% Changing it changes every label, so change it before starting and not during.
TFRAC = 0.02;

% The T onset carries a measured constant on top of the threshold. The reviewer
% does not apply a threshold to the onset at all, they place it at the clear
% change of slope, and that sits consistently later than the first departure from
% baseline. Measured over 270 units in three batches, the gap is +14 ms and it
% does not move between thresholds of 2 and 3 per cent, so it is a property of
% where the reviewer looks rather than an artefact of the threshold.
%
% Taking the earliest departure and adding the constant reproduces the reference
% with an interquartile range near 10 ms. Taking the nearest return before the
% peak, which is what this tool did first, gives a range of 48 to 66 ms on the
% same units. The onset is still the fuzziest landmark here, but it is now fuzzy
% at a tenth of the width.
TON_SHIFT_MS = 14;

% Plausibility ranges for the biomarker readout. A value outside its range
% gets a "!" and is a prompt to look again, not an error. Wide QRS is
% genuine in bundle branch block and long QT is genuine in infarct.
BM = struct('qrs',[40 200], 'qt',[200 700], 'tdur',[60 350], 'tpe',[20 200]);

% ------------------------------------------------------- resolve input(s)
if nargin < 1 || isempty(inputPath)
    [fn, fp] = uigetfile({'*.mat','ECG .mat (single, multi-record, e.g. SimulatedECGs_Smith2026.mat)'}, ...
        'Select an ECG .mat (or Cancel to pick a folder)');
    if isequal(fn,0)
        fp = uigetdir(pwd, 'Select a folder of ECG table .mat files');
        if isequal(fp,0); disp('Cancelled.'); return; end
        inputPath = fp;
    else
        inputPath = fullfile(fp, fn);
    end
end

if exist(inputPath,'dir') == 7
    d = dir(fullfile(inputPath,'*.mat'));
    d = d(~[d.isdir]);
    files = fullfile({d.folder}, {d.name});
    baseDir = inputPath;
elseif exist(inputPath,'file') == 2
    files = {inputPath};
    baseDir = fileparts(inputPath);
else
    error('manual_label_ecg:input','Cannot find "%s".', inputPath);
end
if isempty(files)
    error('manual_label_ecg:nofiles','No .mat files found in "%s".', inputPath);
end

if nargin < 2 || isempty(outDir)
    outDir = fullfile(baseDir, 'labels');
end
if exist(outDir,'dir') ~= 7; mkdir(outDir); end

% -------------------------------------------------------- shared state S
S = struct();
% Each input .mat is expanded into one or more reviewable records: a plain
% single-table / single-struct file gives one record, and a multi-record file
% such as SimulatedECGs_Smith2026.mat (whose variables are cell arrays of ECG
% tables, one cell array per disease class) gives one record per table, tagged
% with its class. buildRecords does the detection.
S.recs      = buildRecords(files);
% Records are browsed through a filtered VIEW, exactly as the Gold tool browses
% its unit list: S.view holds the indices of S.recs that pass the class filter
% and S.pos is the position within that view. S.recIdx is always S.view(S.pos).
S.view       = 1:numel(S.recs);
S.pos        = 1;
S.recIdx     = 1;
S.classFilter = 'all';
% Which records already have a label file on disk, so the list can mark them.
% Computed once here rather than per redraw.
% ---- optional reference-lead worklist -----------------------------------
% If smith2026_worklist.csv sits next to the data it names ONE lead per record
% and the tool switches to worklist mode: opening a record jumps straight to its
% assigned lead, and only that lead is exported. One representative unit per
% record is enough because the four boundaries are shared by all twelve leads
% anyway, so a second lead of the same record adds a correlated observation
% rather than an independent one. Built by build_reference_worklist.m, and it
% contains only records from this dataset.
%
% Delete or rename the file to go back to reviewing all twelve leads.
% Held as plain parent-scope variables rather than fields of S. Nested
% functions see the parent workspace directly, so WL and WLMODE survive every
% `S = guidata(S.fig)` round-trip. Putting them in S made them vulnerable to any
% code path that wrote back a struct built before they were set.
WL = containers.Map('KeyType','char','ValueType','char');
wlPath = '';
for cand = { fullfile(fileparts(mfilename('fullpath')),'smith2026_worklist.csv'), ...
             fullfile(baseDir,'smith2026_worklist.csv'), ...
             fullfile(outDir,'smith2026_worklist.csv'), ...
             fullfile(fileparts(mfilename('fullpath')),'gold_style_worklist.csv') }
    if exist(cand{1},'file') == 2; wlPath = cand{1}; break; end
end
if ~isempty(wlPath)
    try
        W = readtable(wlPath,'TextType','string');
        if all(ismember({'record_id','lead'}, W.Properties.VariableNames))
            for ii = 1:height(W)
                WL(char(W.record_id(ii))) = char(W.lead(ii));
            end
            fprintf('Worklist: %s  (%d units, one lead per record)\n', wlPath, WL.Count);
        end
    catch ME
        warning('Could not read worklist %s: %s', wlPath, ME.message);
    end
end
WLMODE = WL.Count > 0;

% ---- per-lead T mode ----------------------------------------------------
% Held in the parent workspace alongside WL and WLMODE for the same reason: a
% nested function reads it directly and it survives every guidata round-trip.
TLEAD = false;
if nargin >= 3 && ~isempty(mode)
    TLEAD = any(strcmpi(strtrim(char(mode)), {'tlead','t','tperlead','t_per_lead'}));
    if ~TLEAD
        error('manual_label_ecg:mode', ...
            'Unknown mode "%s". The only mode is ''tlead''.', char(mode));
    end
end

% ---- optional record subset ---------------------------------------------
% If smith2026_subset.csv sits beside the data it names the records that are in
% scope, and everything else is hidden. The file is a deliberate design decision
% about which records the external test set contains, so it belongs on disk next
% to the data rather than in a reviewer's head, and the tool simply obeys it.
%
% Hiding rather than deleting is the point. The .mat is the only copy of the
% signals there has ever been, and a record dropped from this pass may still be
% wanted later, so nothing is removed from it.
SUBSET = {};
for cand = { fullfile(fileparts(mfilename('fullpath')),'smith2026_subset.csv'), ...
             fullfile(baseDir,'smith2026_subset.csv'), ...
             fullfile(outDir,'smith2026_subset.csv') }
    if exist(cand{1},'file') == 2
        SUBSET = readIdList(cand{1});
        if ~isempty(SUBSET)
            keepR = ismember({S.recs.name}, SUBSET);
            fprintf('Subset: %s\n  %d of %d records in scope, %d hidden.\n', ...
                    cand{1}, sum(keepR), numel(S.recs), sum(~keepR));
            missing = SUBSET(~ismember(SUBSET, {S.recs.name}));
            if ~isempty(missing)
                warning('%d record(s) named in the subset are not in the data, first is %s', ...
                        numel(missing), missing{1});
            end
            S.recs = S.recs(keepR);
            S.view = 1:numel(S.recs);
            S.pos  = 1;
        end
        break;
    end
end
if isempty(S.recs)
    error('manual_label_ecg:emptySubset', ...
        'The subset file left no records in scope. Check the record names in it.');
end

% The QRS boundaries every lead inherits in tlead mode, read from the combined
% corrections CSV rather than from the per-record .mat. The CSV is the artefact
% the test set is built from, so reading it here guarantees the QRS you see is
% the QRS being scored. Reading the .mat instead would leave the two free to
% drift apart silently, which is the one failure this mode cannot detect.
RECQRS  = containers.Map('KeyType','char','ValueType','any');
RECPEAK = containers.Map('KeyType','char','ValueType','any');
if TLEAD
    corrPath = fullfile(outDir, 'smith2026_manual_corrections.csv');
    [RECQRS, RECPEAK] = readRecordBounds(corrPath);
    if RECQRS.Count == 0
        error('manual_label_ecg:noBounds', ...
            ['tlead mode needs the QRS boundaries that are already agreed.\n' ...
             'Expected them in %s, and found none.\n' ...
             'Run the standard review and apply_boundary_rule first.'], corrPath);
    end
    fprintf(['Per-lead T mode. QRS onset and offset are locked and come from\n' ...
             '  %s  (%d records).\n' ...
             'T start and T end are yours, per lead. Criterion is %g per cent\n' ...
             'of this lead''s own T amplitude above its own baseline.\n'], ...
            corrPath, RECQRS.Count, TFRAC*100);
end

% How many of each record's 12 leads you have explicitly marked checked.
% Counted from the saved label files, so progress survives closing the tool.
S.revCount  = zeros(1,numel(S.recs));
for ii = 1:numel(S.recs)
    lf = fullfile(outDir,[S.recs(ii).name '_labels.mat']);
    if exist(lf,'file') == 2
        try
            prev = load(lf,'L');
            if isfield(prev,'L') && isfield(prev.L,'reviewed')
                if TLEAD
                    % Per-lead T mode counts all twelve again, since every lead
                    % now carries its own T placement and is its own unit. It
                    % counts tReviewed, since reviewed is true on all twelve in
                    % every file written before worklist mode and would report
                    % the whole corpus finished before it had begun.
                    if isfield(prev.L,'tReviewed')
                        S.revCount(ii) = sum([prev.L.tReviewed]);
                    else
                        S.revCount(ii) = 0;
                    end
                elseif WLMODE && isKey(WL, S.recs(ii).name)
                    % Worklist mode counts RECORDS, so a record contributes at
                    % most 1 and only its assigned lead counts. Label files
                    % written under the earlier all-leads scheme have all twelve
                    % leads flagged reviewed; counting those would push the
                    % total past the 162 denominator.
                    kk = find(strcmp(LEADS, WL(S.recs(ii).name)), 1);
                    if ~isempty(kk) && kk <= numel(prev.L)
                        S.revCount(ii) = double(prev.L(kk).reviewed);
                    end
                else
                    S.revCount(ii) = sum([prev.L.reviewed]);
                end
            end
        catch
        end
    end
end
S.LEADS     = LEADS;
S.FIDS      = FIDS;
S.FIDKEY    = FIDKEY;
S.FLABEL    = FLABEL;
S.FIDCOL    = FIDCOL;
S.QGRP      = QGRP;
S.TGRP      = TGRP;
S.CORE      = CORE;
S.MAGLEADS  = MAGLEADS;
S.BM        = BM;
S.WIDTH     = WIDTH;
S.ACTIV     = ACTIVATION;
S.outDir    = outDir;
% Single combined corrections CSV for the whole session, in the Gold reviewer
% column layout. Per-record .mat files sit alongside it and are what the tool
% resumes from; this CSV is the artefact the Delineation scripts consume.
S.outCsv    = fullfile(outDir, 'smith2026_manual_corrections.csv');
S.leadIdx   = 1;
S.selFid    = 1;      % currently selected landmark (for keyboard nudging)
S.dragF     = 0;      % landmark index being dragged (0 = none)
S.tbl       = [];     % current record table (set by loadRecord)
S.M         = [];     % 12 x N matrix of all leads (overlay + magnitude)
S.L         = [];     % current per-lead labels (set by loadRecord)
S.dirty     = false;  % true when there are edits not yet written to disk
S.overlay   = true;   % 12-lead overlay on by default ('o' toggles)
S.magnitude = true;   % spatial magnitude strip on by default ('m' toggles)
S.tangent   = false;  % tangent-method T-offset aid ('g' toggles)
S.xview     = [];     % visible x-limits in ms (zoom / pan)
S.xfit      = [];     % remembered full-beat fit ('f' restores it)
S.msps      = 1;      % ms per sample, read from the record's time column

% ------------------------------------------------------------ build GUI
S.fig = figure('Name','Manual ECG labelling','NumberTitle','off', ...
    'Color','w','Units','normalized','Position',[0.06 0.10 0.86 0.80], ...
    'WindowButtonDownFcn',@onDown, 'WindowButtonMotionFcn',@onMotion, ...
    'WindowButtonUpFcn',@onUp, 'KeyPressFcn',@onKey, ...
    'WindowScrollWheelFcn',@onScroll, 'CloseRequestFcn',@onClose);

% ---- record browser (left), mirroring the Gold tool's filtered unit list ----
% A class popup narrows the list to one disease class, and the list itself picks
% the record. With SimulatedECGs_Smith2026.mat that means you can work through
% (say) the 39 Healthy records without stepping past the infarction ones.
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.010 0.952 0.170 0.030],'String','Records  (filter: class)', ...
    'BackgroundColor','w','FontSize',9,'FontWeight','bold', ...
    'HorizontalAlignment','left');
S.classPop = uicontrol('Style','popupmenu','Parent',S.fig,'Units','normalized', ...
    'Position',[0.010 0.912 0.170 0.030],'String',[{'all'}, uniqueClasses(S.recs)], ...
    'TooltipString','disease class','Callback',@(~,~)onClassFilter());
S.recList = uicontrol('Style','listbox','Parent',S.fig,'Units','normalized', ...
    'Position',[0.010 0.195 0.170 0.705],'FontName','Courier New', ...
    'FontSize',9,'Callback',@(h,~)onRecPick(h));

% lead picker, immediately right of the record list so any lead is one click away
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.188 0.912 0.052 0.030],'String','Lead', ...
    'BackgroundColor','w','FontSize',9,'FontWeight','bold', ...
    'HorizontalAlignment','left');
S.leadList = uicontrol('Style','listbox','Parent',S.fig,'Units','normalized', ...
    'Position',[0.188 0.195 0.052 0.705],'FontName','Courier New', ...
    'FontSize',9,'String',LEADS,'Value',1,'Callback',@(h,~)onLeadPick(h));

% Axes left edge leaves room for the y tick labels and ylabel, which the old
% 0.105 position did not once a record list was added to its left. There is no
% horizontal scroll bar: panning is , and . (or drag-free keyboard/scroll-wheel
% zoom), which keeps the strip below the trace free for the plot itself.
S.ax = axes('Parent',S.fig,'Units','normalized','Position',[0.300 0.255 0.465 0.645]);
box(S.ax,'on'); grid(S.ax,'on'); hold(S.ax,'on');

% ---- right-hand panels -----------------------------------------------------
% Split into a clickable landmark list and a separate readout below it, as in
% the Gold tool. One tall text block could not fit all 26 lines, which is why
% the flag lines at the bottom (inverted T / excluded / edited) were cut off.
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.775 0.870 0.213 0.030],'String','Landmarks (click to select)', ...
    'BackgroundColor','w','FontSize',9,'FontWeight','bold', ...
    'HorizontalAlignment','left');
S.fidList = uicontrol('Style','listbox','Parent',S.fig,'Units','normalized', ...
    'Position',[0.775 0.610 0.213 0.260],'FontName','Courier New', ...
    'FontSize',9.5,'Callback',@(h,~)onFidPick(h));
% Font a touch smaller than the landmark list so the readout's ~14 lines sit
% comfortably inside the panel instead of being clipped at the bottom.
S.bm = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.775 0.195 0.213 0.400],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontName','Courier New','FontSize',9);

% Control buttons, all on one row (mirrors gold_label_ecg.m). The three QRS
% peak toggles are narrow buttons placed inline right after 'T absent', so the
% absent controls read left to right as T, Q, R, S with no separate strip.
% Q, R and S each have no onset or offset of their own, so presence is the
% whole statement about them: a QS complex has no R and no S, an rS complex has
% no Q, and a monophasic R has neither Q nor S. Press once to mark that peak
% absent, press again to restore it from a fresh seed. Q=2, R=3, S=4 in the
% landmark order above.
% Overlay (o), Magnitude (m), Re-seed (r) and Invert T (i) are keyboard-only:
% they are view toggles or per-lead flags used rarely enough that a button for
% each only crowded the row.
%
% There is no SAVE button. Every edit is written to disk as soon as it is made
% (see autoSave), so the only thing a save button could do is repeat work that
% has already happened. 'Del' clears the selected landmark, marking that wave
% absent, and is the button form of the Delete key.
mkbtn (0.012,'Prev lead', @(~,~)changeLead(-1));
mkbtn (0.130,'Next lead', @(~,~)changeLead(+1));
mkbtn (0.248,'T absent',  @(~,~)toggleWave(TGRP));
mkbtnN(0.366,'Q absent',  @(~,~)toggleLandmark(2));
mkbtnN(0.442,'R absent',  @(~,~)toggleLandmark(3));
mkbtnN(0.518,'S absent',  @(~,~)toggleLandmark(4));
mkbtn (0.594,'Del',       @(~,~)clearSel());
mkbtn (0.712,'Exclude',   @(~,~)toggleFlag('exclude'));
% Push this lead's four boundaries onto all twelve. See copyBoundsToAllLeads.
uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
    'Position',[0.830 0.128 0.110 0.055],'String','Bounds->all (b)','FontSize',8.5, ...
    'FontWeight','bold','ForegroundColor',[0.10 0.30 0.70], ...
    'Callback',@(~,~)copyBoundsToAllLeads());

% saved / unsaved status line (updated live as you edit)
S.status = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.012 0.088 0.976 0.028],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',10,'FontWeight','bold', ...
    'ForegroundColor',[0.10 0.50 0.10],'String','All changes saved');

% placement rule, kept on screen because it is the whole point of the strip
S.hint = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.012 0.050 0.976 0.032],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',9.5,'ForegroundColor',[0.60 0.30 0.00], ...
    'String',['Onsets and offsets come from the ORANGE magnitude curve ' ...
     '(first lift off the dotted floor, last return to it).  Peaks come ' ...
     'from the BLACK single-lead trace.']);

% One compact key line. The full protocol is in the header help text (help
% manual_label_ecg), so repeating it on screen only stole height from the plot.
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.012 0.014 0.976 0.028],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',9,'ForegroundColor',[0.40 0.40 0.40], ...
    'String', ...
    ['Nothing to press: edits save themselves, and a record is written when you move off it ' ...
     '(leads you did not touch keep the automatic values).   ' ...
     '1-8 select,  arrows nudge (Shift=5),  Del absent,  ' ...
     'o overlay,  m magnitude,  +/- zoom,  ,/. pan,  f fit,  n/p lead,  Ctrl+n/p record,  ' ...
     'r re-seed,  t T-absent,  i invert-T,  x exclude,  g tangent T-offset aid (Shift+G accepts it),  ' ...
     'b copy this lead''s 4 boundaries to all 12 leads']);

guidata(S.fig, S);
loadRecord();

% ===================================================================
%                          nested functions
% ===================================================================
    function mkbtn(x,label,cb)
        uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
            'Position',[x 0.128 0.110 0.055],'String',label,'FontSize',9, ...
            'Callback',cb);
    end

    % Narrow button variant, so the three QRS-peak absent toggles sit inline on
    % the same row right after 'T absent' (mirrors gold_label_ecg.m).
    function mkbtnN(x,label,cb)
        uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
            'Position',[x 0.128 0.068 0.055],'String',label,'FontSize',8, ...
            'Callback',cb);
    end

    % -------------------------------------------------- load a record
    function loadRecord()
        S = guidata(S.fig);
        if isempty(S.view); return; end
        S.pos    = min(max(1,S.pos), numel(S.view));
        S.recIdx = S.view(S.pos);
        rec = S.recs(S.recIdx);
        recName = rec.name;
        S.recName = recName;
        S.recClass = rec.class;

        % An expanded record already carries its table in memory; a plain
        % single-record file is loaded on demand so a big folder stays light.
        if ~isempty(rec.tbl)
            tbl = rec.tbl;
        else
            raw = load(rec.file);
            tbl = pickTable(raw);
        end
        if isempty(tbl)
            uiwait(errordlg(sprintf('%s contains no MATLAB table.',recName),'Load error'));
            return
        end
        % Normalise the time column to a relative-ms vector and keep a
        % working copy carrying a lowercase 't' column for delineate_ecg_v3.
        vn = tbl.Properties.VariableNames;
        if any(strcmp(vn,'Time'))
            tcol = double(tbl.Time);
        elseif any(strcmp(vn,'t'))
            tcol = double(tbl.t);
        else
            tcol = (0:height(tbl)-1)';   % fall back to 1 ms sampling
        end
        tcol = tcol(:);
        S.trel = tcol - tcol(1);         % ms, relative to beat start
        tbl.t  = S.trel;                 % ensure v3 sees a 't' column
        S.tbl  = tbl;
        S.n    = height(tbl);

        % Sample interval read from the data rather than assumed, so a 1 ms
        % record and a 2 ms record both nudge and save in the right units.
        if S.n > 1
            dt = median(diff(S.trel));
            if ~isfinite(dt) || dt <= 0; dt = 1; end
        else
            dt = 1;
        end
        S.msps = dt;

        % 12 x N matrix once per record, feeding both the overlay and the
        % spatial magnitude curve.
        S.M = nan(numel(S.LEADS), S.n);
        for j = 1:numel(S.LEADS)
            nm = S.LEADS{j};
            if any(strcmp(vn,nm)); S.M(j,:) = double(tbl.(nm)).'; end
        end

        % initialise / resume per-lead labels
        S.L = emptyLabels(S.LEADS, numel(S.FIDKEY));
        lf = fullfile(S.outDir, [recName '_labels.mat']);
        if exist(lf,'file') == 2
            prev = load(lf);
            widthOK = isfield(prev,'L') && numel(prev.L) == numel(S.L) && ...
                all(arrayfun(@(e) numel(e.idx) == numel(S.FIDKEY), prev.L));
            if widthOK
                S.L = prev.L;                 % resume compatible labels
                for i = 1:numel(S.L)
                    if ~isfield(S.L(i),'seeded') || isempty(S.L(i).seeded)
                        S.L(i).seeded = true;
                    end
                end
                % Label files written before per-unit checking existed carry no
                % reviewed field. Default it to false rather than true: those
                % files hold delineator seeds, and calling them checked is
                % exactly the false provenance this flag exists to prevent.
                if ~isfield(S.L,'reviewed')
                    for i = 1:numel(S.L); S.L(i).reviewed = false; end
                end
                if ~isfield(S.L,'seed')
                    for i = 1:numel(S.L); S.L(i).seed = nan(1,numel(S.FIDKEY)); end
                end
                if ~isfield(S.L,'tReviewed')
                    for i = 1:numel(S.L); S.L(i).tReviewed = false; end
                end
            elseif isfield(prev,'L')
                % Saved with a different landmark count (the original
                % six-point set, for instance). Back it up rather than lose
                % it, then carry the six across into their new slots.
                bak = fullfile(S.outDir, [recName '_labels_old.mat']);
                try; movefile(lf, bak); catch; end
                S.L = migrateLabels(prev.L, S.L, S.FIDKEY);
                warning(['Existing labels for %s use a different landmark ' ...
                    'set; the old file is backed up to %s and the six ' ...
                    'shared landmarks were carried across.'], recName, bak);
            end
        end
        % In per-lead T mode, discard the resumed state of any lead you have not
        % yet reviewed and re-seed it. Those leads carry the shared magnitude
        % boundaries copied across by the earlier pass, which is the very
        % convention this pass replaces, so resuming them would start eleven of
        % every twelve leads from the wrong answer. Leads you already reviewed
        % are left exactly as you left them.
        % Per-lead T mode RESUMES. It does not re-seed and it does not compute a
        % starting value for anything.
        %
        % Every lead opens holding the labels the earlier pass left on it, and
        % the reviewer changes what needs changing. That is the right behaviour
        % once every unit is being read by hand. A computed starting value only
        % helps when most units will be accepted untouched, and it carries a real
        % cost when they will not, since a number already sitting on the trace
        % anchors the eye and a lead that is merely stepped past looks reviewed.
        %
        % tReviewed is still defaulted here. It is the progress flag for this
        % pass and it is not a statement about the values.
        if TLEAD && isstruct(S.L) && ~isempty(S.L)
            for i = 1:numel(S.L)
                if ~isfield(S.L(i),'tReviewed') || isempty(S.L(i).tReviewed)
                    S.L(i).tReviewed = false;
                end
            end
        end
        % Which leads of THIS record you have had on screen. Reset per record and
        % set by showLead, so leaving the record can mark exactly what you saw.
        S.seenLead = false(1, numel(S.LEADS));
        % Everything is editable. The QRS still propagates across the twelve
        % leads when you move it, see setFid, which is a correctness requirement
        % rather than a restriction. Press u if you ever want it held still.
        S.qrsUnlock = true;

        S.leadIdx = 1;
        if WLMODE && ~TLEAD && isKey(WL, recName)
            k = find(strcmp(S.LEADS, WL(recName)), 1);
            if ~isempty(k); S.leadIdx = k; end     % open on the assigned lead
        end
        S.selFid  = 1;
        if TLEAD; S.selFid = 6; end   % T start, since QRS is locked
        S.dirty   = false;    % freshly loaded state matches disk
        % Left empty so showLead measures the beat window for this record once
        % the leads are available (see beatWindow).
        S.xfit    = [];
        S.xview   = [];
        guidata(S.fig, S);
        renderRecList();
        renderLeadList();
        showLead();
    end

    % ------------------------------------ left-hand record list + filter
    % The list shows one row per record in the current view, marked '*' once it
    % has a label file on disk, so you can see how far through a class you are.
    function renderRecList()
        S = guidata(S.fig);
        if ~isgraphics(S.recList); return; end
        n = numel(S.view);
        strs = cell(max(n,1),1);
        for i = 1:n
            ri = S.view(i);
            nrl = S.revCount(ri);
            if WLMODE && ~TLEAD; nL = 1; else; nL = numel(S.LEADS); end
            if nrl >= nL
                mk = ' *';                  % all 12 leads checked
            elseif nrl > 0
                mk = sprintf('%2d', nrl);   % partly through this record
            else
                mk = '  ';
            end
            if WLMODE && isKey(WL, S.recs(ri).name)
                strs{i} = sprintf('%s %-22s %s', mk, S.recs(ri).name, WL(S.recs(ri).name));
            else
                strs{i} = sprintf('%s %s', mk, S.recs(ri).name);
            end
        end
        if n == 0; strs = {'(no records match filter)'}; end
        set(S.recList,'String',strs,'Value',min(max(1,S.pos),max(1,n)));
        guidata(S.fig,S);
    end

    function onRecPick(h)
        S = guidata(S.fig);
        v = get(h,'Value');
        if isempty(S.view) || v == S.pos; return; end
        if v > numel(S.view); return; end
        commitRecord();                   % accept the record you are leaving
        S = guidata(S.fig);
        S.pos = v;
        guidata(S.fig,S);
        loadRecord();
    end

    % Rebuild the view for the chosen class. Records keep their own labels on
    % disk, so filtering only changes what is listed, never what is saved.
    function onClassFilter()
        S = guidata(S.fig);
        commitRecord();
        S = guidata(S.fig);
        opts = get(S.classPop,'String');
        S.classFilter = opts{get(S.classPop,'Value')};
        if strcmp(S.classFilter,'all')
            S.view = 1:numel(S.recs);
        else
            keep = arrayfun(@(r) strcmp(classOf(r), S.classFilter), S.recs);
            S.view = find(keep(:)).';
        end
        S.pos = 1;
        guidata(S.fig,S);
        renderRecList();
        if ~isempty(S.view)
            loadRecord();
        else
            cla(S.ax);
            title(S.ax,'(no records match this class filter)');
        end
    end

    % ------------------------------------------- left-hand lead picker
    function renderLeadList()
        S = guidata(S.fig);
        strs = cell(numel(S.LEADS),1);
        for i = 1:numel(S.LEADS)
            mk = '  ';
            if S.L(i).exclude
                mk = ' X';                       % excluded from the test set
            elseif S.L(i).reviewed
                mk = ' *';                       % checked by you
            elseif S.L(i).edited
                mk = ' .';                       % edited but not yet marked
            end
            strs{i} = sprintf('%s %-4s', mk, S.LEADS{i});
        end
        set(S.leadList,'String',strs,'Value',S.leadIdx);
        guidata(S.fig,S);
    end

    function onLeadPick(h)
        S = guidata(S.fig);
        v = get(h,'Value');
        if v == S.leadIdx; return; end
        if S.dirty; saveRecord(true); S = guidata(S.fig); end
        S.leadIdx = v;
        S.selFid  = 1;
        guidata(S.fig,S);
        showLead();
    end

    % -------------------------------------------------- draw one lead
    function showLead()
        S = guidata(S.fig);
        if isempty(S.tbl); return; end
        li   = S.leadIdx;
        lead = S.LEADS{li};
        t    = S.trel;
        V    = double(S.tbl.(lead));
        M    = S.M;

        if ~S.L(li).seeded
            seedLead(li);            % updates S.L, writes guidata
            S = guidata(S.fig);
        end
        % Record that this lead has been on screen. In per-lead T mode leaving
        % the record marks exactly these leads reviewed, so a lead you never
        % opened never enters the export.
        if isfield(S,'seenLead') && li <= numel(S.seenLead) && ~S.seenLead(li)
            S.seenLead(li) = true;
            guidata(S.fig, S);
        end
        idx = S.L(li).idx;

        % ---- default view: the BEAT, not the whole record -----------------
        % These records are 1000 ms long but the beat itself occupies only the
        % first third or so, so fitting the full record wasted most of the axis
        % and pushed the landmarks into a narrow clump. The window is taken from
        % the spatial magnitude curve (where is there any activity at all), so
        % it is the same for every lead of a record and does not depend on the
        % delineator. Only the VIEW changes; every stored sample index is
        % untouched, so the fiducials are identical either way.
        if isempty(S.xfit)
            S.xfit = beatWindow(M, S.LEADS, S.MAGLEADS, t);
            S.xview = S.xfit;
            guidata(S.fig,S);
        end
        if isempty(S.xview); S.xview = S.xfit; end
        xv = S.xview;
        visMask = (t >= xv(1)) & (t <= xv(2));
        if ~any(visMask); visMask = true(size(t)); end

        cla(S.ax); hold(S.ax,'on'); grid(S.ax,'on');

        % ---- twelve-lead overlay, computed BEFORE the y-limits -------------
        % Each other lead is shifted to its own visible median and scaled to the
        % amplitude of the lead being edited, so shape can be compared without
        % the small leads vanishing.
        %
        % The traces are built here rather than at draw time because the y-range
        % has to know about them. Scaling maps each lead onto pC +/- pA, where
        % pA is the reference lead's largest deviation from its median. When the
        % reference is one-sided (aVR, all negative) pC + pA sits far ABOVE the
        % reference's own maximum, so limits taken from the reference alone
        % clipped the top off every overlay trace.
        OVy = {};
        ovLo = Inf; ovHi = -Inf;
        vis = V(visMask);
        if S.overlay
            pC = median(vis,'omitnan');
            pA = max(abs(vis - pC));
            if ~isfinite(pA) || pA == 0; pA = 1; end
            for j = 1:size(M,1)
                if j == li; continue; end
                W = M(j,:);
                if ~any(isfinite(W)); continue; end
                Wv = W(visMask);
                wC = median(Wv,'omitnan');
                wA = max(abs(Wv - wC));
                if ~isfinite(wA) || wA == 0; continue; end
                Wn = (W - wC) * (pA/wA) + pC;
                OVy{end+1} = Wn; %#ok<AGROW>
                wn = Wn(visMask);
                ovLo = min(ovLo, min(wn));
                ovHi = max(ovHi, max(wn));
            end
        end

        % ---- y range: the edited lead AND whatever the overlay adds --------
        lo = min(vis); hi = max(vis);
        if isfinite(ovLo); lo = min(lo, ovLo); end
        if isfinite(ovHi); hi = max(hi, ovHi); end
        ytr = [lo hi];
        if ~all(isfinite(ytr)) || diff(ytr) == 0; ytr = [-1 1]; end
        pad = 0.10*diff(ytr);
        ytr = ytr + [-pad pad];
        yl  = ytr;

        % ---- spatial magnitude strip ------------------------------------
        % Root sum of squares over the eight INDEPENDENT leads, each
        % baseline-removed first. The baseline is taken over the WHOLE beat
        % so the curve keeps its shape as you zoom; the vertical scaling is
        % taken over the VISIBLE window so zooming in still fills the strip.
        if S.magnitude
            gi = find(ismember(S.LEADS, S.MAGLEADS));
            gi = gi(all(isfinite(M(gi,:)), 2));
            if ~isempty(gi)
                D = M(gi, :);
                D = D - median(D, 2, 'omitnan');
                D(~isfinite(D)) = 0;
                mag = sqrt(sum(D.^2, 1));
                magFloor = qtile(mag, 0.10);

                hband = 0.34 * diff(ytr);
                yl(1) = yl(1) - hband;
                mvis = mag(visMask); if isempty(mvis); mvis = mag; end
                mtop = max(mvis); if ~isfinite(mtop) || mtop <= 0; mtop = 1; end
                magY = yl(1) + 0.06*hband + (mag/mtop) * (0.86*hband);
                % divider between the strip and the trace
                plot(S.ax, t([1 end]), (yl(1)+hband)*[1 1], '-', ...
                    'Color',[0.88 0.88 0.90], 'LineWidth',0.5);
                % noise floor, the level an onset must lift off
                fy = yl(1) + 0.06*hband + (magFloor/mtop) * (0.86*hband);
                plot(S.ax, t([1 end]), [fy fy], ':', ...
                    'Color',[0.85 0.42 0.10], 'LineWidth',0.8);
                plot(S.ax, t, min(magY, yl(1)+hband), '-', ...
                    'Color',[0.85 0.42 0.10], 'LineWidth',1.3);
            end
        end

        % ---- draw the overlay traces built above --------------------------
        for j = 1:numel(OVy)
            plot(S.ax, t, OVy{j}, '-', 'Color',[0.62 0.66 0.72], 'LineWidth',0.6);
        end

        % ---- the lead being edited --------------------------------------
        plot(S.ax, t, V, '-', 'Color',[0 0 0], 'LineWidth',1.6);

        % ---- tangent-method T offset (the 'g' aid) ------------------------
        % Draws the construction your Gold protocol specifies: the steepest
        % descending limb after the T peak, extended to the isoelectric
        % baseline. Where it crosses is the T offset. It is drawn, never
        % applied, so it stays an aid rather than another automatic guess -
        % Shift+G accepts it if you agree with it.
        if isfield(S,'tangent') && S.tangent
            TT = tangentOffset(V, idx(7), S.n);
            if TT.ok
                xc = (TT.cross-1) * S.msps;
                x0 = (TT.i0   -1) * S.msps;
                % baseline the construction is measured against
                plot(S.ax, [xv(1) xv(2)], TT.base*[1 1], ':', ...
                     'Color',[0.45 0.45 0.50], 'LineWidth',0.8);
                % the tangent itself, drawn a little either side of the contact
                span = abs(xc - x0);
                xa = x0 - 0.25*span;  xb = xc + 0.12*span;
                ya = TT.base + (TT.slope/max(S.msps,eps))*(xa - xc);
                yb = TT.base + (TT.slope/max(S.msps,eps))*(xb - xc);
                plot(S.ax, [xa xb], [ya yb], '-', ...
                     'Color',[0.85 0.15 0.55], 'LineWidth',1.4);
                % contact point on the steepest limb
                plot(S.ax, x0, V(min(max(1,round(TT.i0)),S.n)), 'o', ...
                     'MarkerSize',6, 'LineWidth',1.4, ...
                     'MarkerEdgeColor',[0.85 0.15 0.55], 'MarkerFaceColor','w');
                % the crossing = the tangent-method T offset
                plot(S.ax, [xc xc], yl, '--', ...
                     'Color',[0.85 0.15 0.55], 'LineWidth',1.6);
                text(xc, yl(1) + 0.06*diff(yl), sprintf(' tangent %.0f ms ', xc), ...
                     'Parent',S.ax, 'Color',[0.85 0.15 0.55], 'FontSize',8.5, ...
                     'FontWeight','bold', 'BackgroundColor',[1 1 1], ...
                     'EdgeColor',[0.85 0.15 0.55], 'Margin',1, ...
                     'HorizontalAlignment','left', 'VerticalAlignment','middle', ...
                     'Clipping','on');
            end
        end

        % ---- landmarks, with staggered boxed labels ---------------------
        % Six tiers keep neighbouring labels off each other when QRS end,
        % T start and the peaks crowd together.
        S.hMark = gobjects(1,numel(S.FIDS));
        S.hLine = gobjects(1,numel(S.FIDS));
        yr = diff(ytr); nT = 4; shown = 0;
        for k = 1:numel(S.FIDS)
            xi = idx(k);
            if ~isfinite(xi); continue; end
            xi = min(max(1,round(xi)), S.n);
            xk = t(xi);
            lwk = 1.0; fw = 'normal';
            if k == S.selFid; lwk = 2.0; fw = 'bold'; end
            S.hLine(k) = plot(S.ax, [xk xk], yl, '-', ...
                'Color',S.FIDCOL(k,:), 'LineWidth',lwk);
            S.hMark(k) = plot(S.ax, xk, V(xi), 'o', ...
                'MarkerSize',9, 'LineWidth',1.5, ...
                'MarkerEdgeColor',S.FIDCOL(k,:), 'MarkerFaceColor','w');
            if k == S.selFid
                set(S.hMark(k),'MarkerSize',12,'LineWidth',2.6);
            end
            tier = mod(shown, nT); shown = shown + 1;
            ylab = ytr(2) - 0.04*yr - tier*(0.92*yr/nT);
            text(xk, ylab, [' ' S.FLABEL{k} ' '], 'Parent',S.ax, ...
                'Color',S.FIDCOL(k,:), 'FontSize',8.5, 'FontWeight',fw, ...
                'BackgroundColor',[1 1 1], 'EdgeColor',S.FIDCOL(k,:), ...
                'Margin',1, 'LineWidth',0.5, 'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', 'Clipping','on');
        end

        ylim(S.ax, yl); xlim(S.ax, xv);
        xlabel(S.ax,'time (ms, relative to beat start)');
        ylabel(S.ax,'voltage (normalised)');

        flagStr = '';
        if S.L(li).exclude;   flagStr = [flagStr '  [EXCLUDED]']; end
        if S.L(li).invertedT; flagStr = [flagStr '  [inverted T]']; end
        clsStr = '';
        if isfield(S,'recClass') && ~isempty(S.recClass)
            clsStr = ['   -   ' strrep(upper(S.recClass),'_','\_')];
        end
        title(S.ax, sprintf('%s%s   |   lead %s (%d/%d)   |   rec %d/%d   |   %.3g ms/sample%s', ...
            strrep(S.recName,'_','\_'), clsStr, lead, li, numel(S.LEADS), ...
            S.pos, numel(S.view), S.msps, flagStr), ...
            'FontSize',11);

        guidata(S.fig, S);   % persist handles before any callback reads them
        updateInfo(); refreshStatus();
    end

    % --------------------------- seed one lead from delineate_ecg_v3
    function seedLead(li)
        S = guidata(S.fig);
        lead = S.LEADS{li};
        V   = double(S.tbl.(lead));
        idx = nan(1, numel(S.FIDKEY));
        idx(S.CORE) = defaultIdx(S.n);
        inv = false;
        try
            [qs,qe,~,~,~,~,~,tp,~,ts,te,inverse] = ...
                delineate_ecg_v3(S.tbl, lead, S.WIDTH, S.ACTIV, S.n);
            if all(isfinite([qs qe ts tp te]))
                idx(1) = qs; idx(5) = qe;
                idx(6) = ts; idx(7) = tp; idx(8) = te;
                [q,r,s] = nameQRS(V, qs, qe);
                idx(2) = q; idx(3) = r; idx(4) = s;
                inv = logical(inverse);
            end
        catch ME
            warning('Auto-seed failed for lead %s (%s); using defaults.', ...
                lead, ME.message);
        end
        % In per-lead T mode the seed is not the delineator's guess. QRS comes
        % from the agreed record-level boundaries and must be identical in all
        % twelve leads, and the T boundaries start at this lead's own return to
        % baseline rather than at a v3 value that was never per-lead. The peaks
        % are still named against the QRS window, so they are re-derived after
        % the window is replaced rather than before.
        v3idx = idx;      % keep the untouched delineator output for `seed`
        if TLEAD
            qb = recordQrs();
            if all(isfinite(qb))
                idx(1) = qb(1); idx(5) = qb(2);
                [q,r,s] = nameQRS(V, qb(1), qb(2));
                idx(2) = q; idx(3) = r; idx(4) = s;
            end
            % The T is left as the delineator produced it. This branch now runs
            % only for a lead with no saved state at all, which on a corpus that
            % has already been through a first pass should never happen, and
            % computing a T here would be putting a starting value under a
            % reviewer who is reading every unit anyway.
            % Put the reviewer's own peaks back where the corrections CSV holds
            % them. Only the nominated lead of each record has a row, so this
            % affects that lead alone, and it is exactly the lead whose peaks
            % were placed by hand. A blank cell means the reviewer marked the
            % wave absent, which is a decision and is carried through as NaN.
            pk = leadPeaks(lead);
            if ~isempty(pk)
                idx(2) = pk(1); idx(3) = pk(2); idx(4) = pk(3); idx(7) = pk(4);
            end
        end
        S.L(li).idx       = clampIdx(idx, S.n);
        % The automatic values BEFORE any correction. Edits change idx and never
        % touch seed, so (idx - seed) is the delineator's signed error on a
        % human-checked unit, which is the quantity this exercise estimates.
        %
        % In per-lead T mode the STARTING POINT is not the delineator output, so
        % the two part company. seed keeps the delineator, since that is what the
        % v3_* columns claim to hold and what the signed-error estimate needs.
        % Overwriting it with the tlead starting point would silently turn every
        % v3 error into zero.
        S.L(li).seed      = clampIdx(v3idx, S.n);
        S.L(li).invertedT = inv;
        S.L(li).seeded    = true;
        S.L(li).edited    = false;
        S.L(li).source    = 'v3-seed';
        if TLEAD; S.L(li).source = 'tlead-seed'; end
        guidata(S.fig, S);
    end

    % The agreed QRS boundaries for the current record, 1-based, from the
    % corrections CSV. Empty when the record is not in the file, which in tlead
    % mode means it was never reviewed and has no boundaries to inherit.
    function qb = recordQrs()
        qb = [NaN NaN];
        if ~TLEAD || ~isfield(S,'recName') || isempty(S.recName); return; end
        if ~isKey(RECQRS, S.recName); return; end
        v  = RECQRS(S.recName);          % 0-based on disk
        qb = [v(1)+1, v(2)+1];
    end

    % The hand-placed peaks for one lead of the current record, 1-based, or empty
    % when the corrections CSV has no row for it. A row that exists but left a
    % peak blank returns NaN in that slot, which is the reviewer's absent call
    % and must survive rather than be re-derived.
    function pk = leadPeaks(lead)
        pk = [];
        if ~TLEAD || ~isfield(S,'recName') || isempty(S.recName); return; end
        key = [S.recName '|' lead];
        if ~isKey(RECPEAK, key); return; end
        v = RECPEAK(key);
        pk = v + 1;                       % 0-based on disk
    end

    % T onset, T peak and T offset read from ONE lead's own trace.
    %
    % The boundary sits where that lead's trace falls within TFRAC of the T
    % amplitude of the lead's own isoelectric baseline. Referencing the threshold
    % to the wave's own amplitude is what makes a lead with a small T and a lead
    % with a large one answer the same question, and it is what the measurement
    % behind TFRAC was made on.
    %
    % The baseline is the median of the record's quiet tail. These records hold a
    % single beat with several hundred milliseconds of flat signal after the T,
    % so that tail is genuinely isoelectric. On multi-beat data it would not be,
    % and the pre-QRS segment would have to be used instead.
    %
    % This is deliberately NOT the 5 per cent of the MAGNITUDE rule that the
    % shared readout uses, on two counts. It reads this lead's own trace rather
    % than a curve pooled over eight leads, and it is referenced to the T
    % amplitude rather than to the magnitude's noise floor. Both differences are
    % the point. See SMITH2026_TLEAD_ADDENDUM.md.
    %
    % Absolute deviation is used throughout, so an inverted T needs no special
    % case. Returns NaN for anything it cannot establish rather than a guess.
    function [tOn, tPk, tOff] = leadTBounds(V, qOff)
        tOn = NaN; tPk = NaN; tOff = NaN;
        V = double(V(:)).';
        nn = numel(V);
        if nn < 20 || ~isfinite(qOff); return; end
        b0   = max(1, round(0.85*nn));
        tail = V(b0:nn); tail = tail(isfinite(tail));
        if numel(tail) < 5; return; end
        base = median(tail);

        D  = abs(V - base);
        qi = min(nn, max(1, round(qOff) + 1));
        lo = min(nn, qi + 10);                       % clear of the J point
        if lo >= nn - 2; return; end
        [pk, rel] = max(D(lo:nn));
        tPk = lo + rel - 1;
        if ~isfinite(pk) || pk <= 0; tPk = NaN; return; end
        thr = TFRAC * pk;                 % TFRAC of this lead's own T amplitude

        % Offset: the LAST sample beyond the threshold, not the first return to
        % it. The two differ whenever the trace crosses the baseline and comes
        % back, which happens on any T with a terminal overshoot of opposite
        % polarity, and the first return then cuts the wave at the crossing and
        % leaves a visible deflection outside the labelled wave.
        %
        % Which one the reference uses was measured rather than assumed. Across
        % 270 human placements the largest excursion left beyond the reviewer's
        % own T offset has a median of about 1 per cent of the T amplitude and a
        % 95th centile under 5, so the reviewer leaves essentially nothing after
        % it. Scored against those placements the last return gives a median
        % difference of 0 to 4 ms and the first return 4 ms, so the last return
        % is at least as faithful and it is the one that survives an overshoot.
        i = nn;
        while i > tPk && D(i) <= thr; i = i - 1; end
        tOff = min(nn, i + 1);

        % Onset: the EARLIEST departure beyond the threshold after the J point,
        % plus the measured constant. Where the ST is fused with the T the trace
        % never comes inside the threshold at all and the departure is the J
        % point itself, which is the honest answer for a beat with no observable
        % ST segment.
        j = qi;
        while j < tPk && D(j) <= thr; j = j + 1; end
        tOn = max(qi, j - 1) + round(TON_SHIFT_MS / max(S.msps, eps));
        tOn = min(tOn, tPk);
    end

    % Accept the whole record as it currently stands and write it out.
    %
    % Called when you navigate AWAY from a record, never when you arrive at one.
    % That distinction is the whole design: arriving proves nothing, whereas
    % moving on means you had the record in front of you and are content with
    % it. Leads you corrected carry your values; leads you left alone carry the
    % automatic seed, which is you accepting it. Either way nothing is asked of
    % you and no button is pressed.
    %
    % Records you never open are still never written, so the export cannot
    % contain delineator output you never looked at.
    function commitRecord()
        S = guidata(S.fig);
        if isempty(S.tbl) || ~isstruct(S.L) || isempty(S.L); return; end
        for li = 1:numel(S.LEADS)
            if ~S.L(li).seeded; seedLead(li); S = guidata(S.fig); end
        end
        if TLEAD
            % Per-lead T mode marks exactly the leads you actually displayed.
            %
            % The all-leads rule below ("leaving the record means you accepted
            % all twelve") cannot apply here, since only one lead is ever on
            % screen at a time and eleven of the twelve would be asserted
            % reviewed without having been shown. The worklist rule cannot apply
            % either, since every lead now carries its own T placement and is its
            % own unit. Displaying a lead and moving on IS accepting it, so
            % showLead records the visit and this marks what was visited.
            any_new = false;
            if ~isfield(S,'seenLead'); S.seenLead = false(1,numel(S.LEADS)); end
            for li = 1:numel(S.LEADS)
                if li <= numel(S.seenLead) && S.seenLead(li) && ~S.L(li).tReviewed
                    S.L(li).tReviewed = true;
                    S.L(li).reviewed  = true;
                    any_new = true;
                end
            end
            if ~any_new && ~S.dirty; return; end
        elseif WLMODE
            % Only the assigned unit counts. Marking the other eleven would put
            % automatic output into the test set dressed as reviewed labels.
            k = [];
            if isKey(WL, S.recName)
                k = find(strcmp(S.LEADS, WL(S.recName)), 1);
            end
            if isempty(k); k = S.leadIdx; end
            if S.L(k).reviewed && ~S.dirty; return; end
            S.L(k).reviewed = true;
        else
            if all([S.L.reviewed]) && ~S.dirty; return; end
            for li = 1:numel(S.LEADS); S.L(li).reviewed = true; end
        end
        S.dirty = true;
        guidata(S.fig,S);
        saveRecord(true);
    end

    % Copy THIS lead's four boundaries onto all twelve leads of the record.
    %
    % A wave boundary is a property of the heartbeat, not of one lead: the true
    % onset is the earliest instant ANY lead leaves baseline and the true offset
    % the latest instant any lead returns to it. A lead whose axis sits near
    % perpendicular to the wavefront simply registers that instant weakly, which
    % makes its own apparent boundary late at the onset and early at the offset.
    % Placing a boundary independently per lead therefore bakes that projection
    % error into the labels rather than removing it.
    %
    % The training corpus follows the same convention. Across the twelve leads
    % of a record in pretrain_units.csv, qrs_onset, qrs_offset, t_onset and
    % t_offset each hold exactly ONE distinct value, while q, r, s and t_peak
    % vary lead by lead. So peaks stay per-lead here and only the boundaries are
    % propagated.
    %
    % Decide the four boundaries once on whichever lead shows them most clearly,
    % reading them off the magnitude curve with the overlay on, then press b.
    function copyBoundsToAllLeads()
        copyBounds([1 5 6 8], 'Boundaries');
    end

    % QRS onset and offset only. This is what 'b' does in per-lead T mode, and
    % it is available on Shift+B everywhere else. Copying all four would undo
    % every per-lead T placement in the record in one keystroke, which is the
    % single most expensive mistake available in that mode, so the destructive
    % version is not reachable there at all.
    function copyQrsToAllLeads()
        copyBounds([1 5], 'QRS boundaries');
    end

    function copyBounds(B, what)
        S = guidata(S.fig);
        if isempty(S.tbl) || ~isstruct(S.L) || isempty(S.L); return; end
        % Refused in per-lead T mode wherever it is called from, the key or the
        % on-screen button. QRS is already shared and locked there, and copying
        % T across would undo a record's worth of per-lead work in one click.
        if TLEAD
            set(S.status,'ForegroundColor',[0.75 0.10 0.10], 'String', ...
                ['Bounds->all is disabled in per-lead T mode. QRS is already ' ...
                 'shared and locked, and T is per lead by design.']);
            return;
        end
        for li = 1:numel(S.LEADS)                 % seed first: seedLead resets
            if ~S.L(li).seeded                    % edited, so it must run before
                seedLead(li); S = guidata(S.fig); % the flags are set below
            end
        end
        vals = S.L(S.leadIdx).idx(B);
        for li = 1:numel(S.LEADS)
            S.L(li).idx(B)   = vals;
            S.L(li).edited   = true;
            S.L(li).reviewed = true;
            S.L(li).source   = 'manual';
        end
        S.dirty = true;
        guidata(S.fig,S);
        showLead(); renderLeadList(); autoSave();
        S = guidata(S.fig);
        set(S.status,'ForegroundColor',[0.10 0.30 0.70], 'String', ...
            sprintf('%s from lead %s copied to all %d leads (peaks left alone)', ...
                    what, S.LEADS{S.leadIdx}, numel(S.LEADS)));
    end

    function reseedLead()
        S = guidata(S.fig);
        S.L(S.leadIdx).seeded = false;
        S.dirty = true;
        guidata(S.fig, S);
        showLead(); renderLeadList(); autoSave();
    end

    % -------------------------------------------------- info readout
    % The landmark times go in their own clickable list and the derived readout
    % goes in the panel below it, so neither can push the other off screen. This
    % is what fixes the cut-off inverted-T / excluded / edited lines.
    function updateInfo()
        S = guidata(S.fig);
        li = S.leadIdx; idx = S.L(li).idx; t = S.trel;
        ms = @(k) fidMs(idx, k, t);

        items = cell(numel(S.FIDS),1);
        for k = 1:numel(S.FIDS)
            v = ms(k);
            if isnan(v)
                items{k} = sprintf('%-9s   absent', S.FIDS{k});
            else
                items{k} = sprintf('%-9s %6.0f ms', S.FIDS{k}, v);
            end
        end
        set(S.fidList,'String',items,'Value',min(max(1,S.selFid),numel(items)));

        qrs  = ivInterval(ms(5), ms(1));   % QRS end  - QRS start
        qt   = ivInterval(ms(8), ms(1));   % T end    - QRS start
        twd  = ivInterval(ms(8), ms(6));   % T end    - T start
        tpe  = ivInterval(ms(8), ms(7));   % T end    - T peak
        lines = {};
        % In per-lead T mode this block goes FIRST. The panel is a fixed height
        % and the readout already fills it, so anything appended at the bottom is
        % clipped and the reviewer never sees the one criterion they are working
        % to. The QRS biomarkers below are unchanged but they are not what this
        % pass is placing.
        if TLEAD
            Vl = double(S.tbl.(S.LEADS{li}));
            [pOn, ~, pOff] = leadTBounds(Vl, idx(5));
            lines{end+1} = 'reference only, not applied';
            lines{end+1} = sprintf('lead %g%% Ton : %4.0f ms', TFRAC*100, (pOn-1)*S.msps);
            c6 = ms(6);
            if ~isnan(c6) && isfinite(pOn)
                lines{end+1} = sprintf('  T start is: %+4.0f ms', c6 - (pOn-1)*S.msps);
            end
            lines{end+1} = sprintf('lead %g%% Toff: %4.0f ms', TFRAC*100, (pOff-1)*S.msps);
            c8 = ms(8);
            if ~isnan(c8) && isfinite(pOff)
                lines{end+1} = sprintf('  T end is  : %+4.0f ms', c8 - (pOff-1)*S.msps);
            end
            BB = [1 5]; sp = 0;
            for b = BB
                vv = arrayfun(@(e) e.idx(b), S.L);
                vv = vv(isfinite(vv));
                if numel(vv) > 1; sp = max(sp, (max(vv)-min(vv))*S.msps); end
            end
            if isfield(S,'qrsUnlock') && ~S.qrsUnlock
                lines{end+1} = 'QRS held (u)';
            elseif sp == 0
                lines{end+1} = 'QRS agrees  : yes';
            else
                lines{end+1} = sprintf('QRS SPREAD  : %.0f ms  (should be 0)', sp);
            end
            lines{end+1} = sprintf('T reviewed  : %d of %d leads', ...
                sum([S.L.tReviewed]), numel(S.LEADS));
            lines{end+1} = '';
        end
        lines{end+1} = bmLine('QRS',   qrs, S.BM.qrs(1),  S.BM.qrs(2));
        lines{end+1} = bmLine('QT',    qt,  S.BM.qt(1),   S.BM.qt(2));
        lines{end+1} = bmLine('T dur', twd, S.BM.tdur(1), S.BM.tdur(2));
        lines{end+1} = bmLine('Tp-Te', tpe, S.BM.tpe(1),  S.BM.tpe(2));
        lines{end+1} = '';
        lines{end+1} = sprintf('Q R S T    : %d %d %d %d', ...
            double(isfinite(idx(2))), double(isfinite(idx(3))), ...
            double(isfinite(idx(4))), double(any(isfinite(idx(S.TGRP)))));
        lines{end+1} = '';
        lines{end+1} = sprintf('inverted T : %d', S.L(li).invertedT);
        lines{end+1} = sprintf('excluded   : %d', S.L(li).exclude);
        lines{end+1} = sprintf('edited     : %d', S.L(li).edited);
        % tangent-method T offset, and how far the current marker sits from it
        if isfield(S,'tangent') && S.tangent
            Vt = double(S.tbl.(S.LEADS{li}));
            TT = tangentOffset(Vt, idx(7), S.n);
            lines{end+1} = '';
            if TT.ok
                xc = (TT.cross-1) * S.msps;
                lines{end+1} = sprintf('tangent Toff: %4.0f ms', xc);
                cur = ms(8);
                if ~isnan(cur)
                    lines{end+1} = sprintf('  T end is  : %+4.0f ms', cur - xc);
                end
            else
                lines{end+1} = 'tangent Toff: n/a';
            end
        end
        % T offset suggested by the spatial magnitude at a fixed 5 percent of the
        % T hump's own height above the noise floor. Measured on these records,
        % this is the criterion that matches the data: their T waves end with a
        % long slow tail, so the amplitude keeps falling well after the steep
        % limb, and the answer is very sensitive to where you cut. Using ONE
        % threshold for every unit is what makes the residual a property of the
        % delineator rather than of the reviewer.
        [mtOn, mtOff, fused, mqOn, mqOff] = magTonToff(); %#ok<ASGLU>
        if isfinite(mqOff)
            lines{end+1} = '';
            lines{end+1} = sprintf('mag 5%% QRSoff: %4.0f ms', mqOff);
            cur5 = ms(5);
            if ~isnan(cur5)
                lines{end+1} = sprintf('  QRS end is: %+4.0f ms', cur5 - mqOff);
            end
        end
        if isfinite(mtOn)
            lines{end+1} = '';
            if fused
                lines{end+1} = sprintf('mag 5%% Ton  : %4.0f ms  FUSED', mtOn);
            else
                lines{end+1} = sprintf('mag 5%% Ton  : %4.0f ms', mtOn);
            end
            cur6 = ms(6);
            if ~isnan(cur6)
                lines{end+1} = sprintf('  T start is: %+4.0f ms', cur6 - mtOn);
            end
        end
        if isfinite(mtOff)
            lines{end+1} = sprintf('mag 5%% Toff : %4.0f ms', mtOff);
            cur8 = ms(8);
            if ~isnan(cur8)
                lines{end+1} = sprintf('  T end is  : %+4.0f ms', cur8 - mtOff);
            end
        end

        % Cross-lead boundary agreement. Only meaningful when every lead is being
        % reviewed: in worklist mode eleven of the twelve still hold automatic
        % seeds, so the spread would measure the seeds and not your work. In
        % per-lead T mode the equivalent line is at the top of the panel and
        % covers the QRS alone, since T disagreeing across leads is the quantity
        % that pass exists to record.
        if ~WLMODE && ~TLEAD
            BB = [1 5 6 8]; sp = 0;
            for b = BB
                vv = arrayfun(@(e) e.idx(b), S.L);
                vv = vv(isfinite(vv));
                if numel(vv) > 1; sp = max(sp, (max(vv)-min(vv))*S.msps); end
            end
            lines{end+1} = '';
            if sp == 0
                lines{end+1} = 'bounds agree : yes';
            else
                lines{end+1} = sprintf('bounds spread: %.0f ms  (b copies)', sp);
            end
        end
        lines{end+1} = '';
        lines{end+1} = sprintf('view %.0f-%.0f ms', S.xview(1), S.xview(2));
        % ordering check across whatever landmarks are present
        seq = idx(isfinite(idx));
        if any(diff(seq) < 0)
            lines{end+1} = '';
            lines{end+1} = 'WARNING: out of order';
        end
        set(S.bm,'String',lines);
    end

    % Both T boundaries from the spatial magnitude.
    %
    % T OFFSET is taken at 5 per cent of the T hump's height above the record's
    % NOISE FLOOR. The T ends by decaying back toward baseline, so the floor is
    % the right thing to measure against.
    %
    % T ONSET is taken at 5 per cent of the T's rise above the ST LEVEL, where
    % the ST level is the lowest the magnitude reaches between QRS offset and the
    % T peak. Anchoring the onset on the noise floor instead was wrong, and
    % measurably so: when the T is small relative to the QRS - about 0.2 to 0.3
    % on these records - 5 per cent of the T height is a tiny absolute value, so
    % any residual ST deviation keeps the curve above it all the way back to the
    % QRS and the rule collapses to "T onset = QRS offset" every time. Anchoring
    % on the ST level asks the question that actually matters: where does the T
    % start rising off whatever it is sitting on.
    %
    % The two agree when the ST segment is genuinely isoelectric, because then
    % the ST level IS the floor. So this degrades gracefully to the simple rule
    % on the healthy records while staying meaningful on the fused ones.
    %
    % FUSED reports that the magnitude never returns to floor level between QRS
    % offset and the T peak - there is no isoelectric ST segment. It is a
    % description of the record, not a fault, and it does not change how the
    % onset is computed.
    function [vOn, vOff, fused, qOn, qOff] = magTonToff()
        vOn = NaN; vOff = NaN; fused = false; qOn = NaN; qOff = NaN;
        S = guidata(S.fig);
        if isempty(S.tbl) || ~isfield(S,'M') || isempty(S.M); return; end
        gi = find(ismember(S.LEADS, S.MAGLEADS));
        gi = gi(gi <= size(S.M,1));
        gi = gi(all(isfinite(S.M(gi,:)), 2));
        if numel(gi) < 4; return; end
        D = S.M(gi,:);
        D = D - median(D, 2, 'omitnan');
        D(~isfinite(D)) = 0;
        mag = sqrt(sum(D.^2, 1));
        nn  = numel(mag);
        fl  = qtile(mag, 0.10);
        % QRS boundaries straight off the magnitude, at 5 per cent of the QRS
        % height above the floor: walk out from the tallest point in both
        % directions. These are what the reviewer was previously having to
        % eyeball.
        [qpk, ipk] = max(mag);
        qamp = qpk - fl;
        if isfinite(qamp) && qamp > 0
            thrQ = fl + 0.05*qamp;
            % Onset by scanning FORWARD for the first lift off the floor rather
            % than walking back from the QRS peak. The back-walk stops at the
            % first sub-threshold sample going backwards, so on a complex with a
            % separated Q lobe it stops in the dip between Q and R and reports
            % the onset AFTER the Q - 22 ms late on AnteriorInfarction_013.
            kOn = find(mag > thrQ, 1, 'first');
            if isempty(kOn); kOn = ipk; end
            qOn  = (max(1, kOn-1) - 1) * S.msps;
            % Walking forward, STOP at the trough between the QRS and the T as
            % well as at the threshold. Threshold alone was wrong: when the T is
            % large relative to the QRS - which is exactly the ischemia case -
            % the curve never drops to 5 per cent of the QRS height between the
            % two, so the walk ran straight through the T wave and reported a
            % QRS offset hundreds of ms late. The QRS ends at the quiet point
            % before the T, whether or not that point is deep.
            % Locate the T peak first, searching only from 150 ms onward - in
            % this dataset every QRS is over by 135 ms and every T peak falls
            % after 210 ms - then bound the walk at the LOWEST point between
            % the QRS peak and the T peak. Taking the first local minimum
            % instead was wrong: a biphasic QRS puts a notch on the magnitude,
            % the walk stopped inside the notch, and QRS offset came out 30-40
            % ms early on 8 of the 162 records.
            jstop = nn;
            t0 = min(nn, max(ipk + 20, round(150/S.msps) + 1));
            if t0 < nn
                [~, r0] = max(mag(t0:nn));
                tp0 = t0 + r0 - 1;
                if tp0 > ipk + 2
                    [~, r1] = min(mag(ipk:tp0));
                    jstop = ipk + r1 - 1;
                end
            end
            i = ipk;
            while i < jstop && mag(i) > thrQ; i = i + 1; end
            qOff = (i-1) * S.msps;
        end

        qo  = S.L(S.leadIdx).idx(5);
        if ~isfinite(qo); return; end
        qi = min(max(1, round(qo)+1), nn);
        lo = min(nn, qi + 10);
        if lo >= nn - 2; return; end
        [pkv, rel] = max(mag(lo:nn));
        tp  = lo + rel - 1;
        amp = pkv - fl;
        if ~isfinite(amp) || amp <= 0; return; end

        % ---- T offset: 5 per cent of the T height above the noise floor ----
        i = tp; while i < nn && mag(i) > fl + 0.05*amp; i = i + 1; end
        vOff = (i-1) * S.msps;

        % ---- T onset: 5 per cent of the T's rise above the ST level --------
        if tp <= qi; return; end
        seg = mag(qi:tp);
        [stLev, jrel] = min(seg);
        jmin  = qi + jrel - 1;                 % the knee: lowest point of the ST
        amp2  = pkv - stLev;
        if ~isfinite(amp2) || amp2 <= 0; vOn = (jmin-1)*S.msps; return; end
        thr2  = stLev + 0.05*amp2;
        i = jmin; while i < tp && mag(i) < thr2; i = i + 1; end
        vOn   = (i-1) * S.msps;

        % Is there a real isoelectric ST segment? Test the ST minimum against the
        % record's own NOISE level, not against a fraction of the T height. The
        % latter is a tiny threshold whenever the T is small, so it flagged
        % FUSED even on records with an obvious flat ST - which was wrong and
        % misleading. Noise is taken from the quiet tail of the record.
        b0 = max(1, round(0.85*nn));
        tl = mag(b0:nn); tl = tl(isfinite(tl));
        if numel(tl) > 5
            sd = std(tl);
            fused = (stLev - median(tl)) > max(5*sd, 0.02*amp);
        else
            fused = (stLev - fl) > 0.05*amp;
        end
    end

    % Clicking a landmark in the right-hand list selects it, so the arrow keys
    % and 'click the trace to move it' work without reaching for the 1-8 keys.
    function onFidPick(h)
        S = guidata(S.fig);
        S.selFid = get(h,'Value');
        guidata(S.fig,S);
        showLead();
    end

    % --------------------------------------------- move a landmark
    function setFid(k, newIdx)
        if qrsLocked(k); return; end
        S = guidata(S.fig);
        li = S.leadIdx;
        newIdx = min(max(1,round(newIdx)), S.n);
        S.L(li).idx(k) = newIdx;
        S.L(li).edited   = true;
        S.L(li).reviewed = true;   % an edit IS a review; no button needed
        if TLEAD; S.L(li).tReviewed = true; end
        S.L(li).source   = 'manual';
        % A QRS correction in per-lead T mode belongs to the beat, so it goes to
        % every lead at once. Doing it per lead would leave the record carrying
        % two QRS answers, which the QRS SPREAD line would report but which
        % nothing downstream is built to resolve.
        if TLEAD && any(k == [1 5])
            for jj = 1:numel(S.LEADS)
                S.L(jj).idx(k) = newIdx;
                S.L(jj).edited = true;
            end
            S.qrsEdited = true;
            % Keep the record's inherited QRS in step with the correction.
            % RECQRS is what seedLead reads, so without this a later re-seed of
            % any lead would quietly restore the value you have just rejected,
            % and the T window would be measured from it again.
            RECQRS(S.recName) = [S.L(1).idx(1)-1, S.L(1).idx(5)-1];
        end
        S.dirty = true;
        guidata(S.fig, S);
        showLead();
        % Persist straight away, EXCEPT while a marker is being dragged: a drag
        % fires setFid on every mouse-move, so saving here would rewrite the CSV
        % dozens of times per gesture. onUp saves once when the drag finishes.
        S = guidata(S.fig);
        if S.dragF == 0; autoSave(); end
    end

    % QRS onset and QRS offset are read-only in per-lead T mode. They are the
    % agreed record-level values the test set already carries, they are the same
    % in all twelve leads by construction, and this pass is not reopening them.
    % Blocking the edit at the single point every change funnels through is what
    % makes that guarantee hold for a drag, an arrow key and a click alike.
    function tf = qrsLocked(k)
        tf = false;
        if ~TLEAD; return; end
        if ~any(k == [1 5]); return; end
        S = guidata(S.fig);
        if ~isfield(S,'qrsUnlock') || S.qrsUnlock; return; end
        tf = true;
        if isfield(S,'status') && isgraphics(S.status)
            set(S.status,'ForegroundColor',[0.75 0.10 0.10], 'String', ...
                'QRS is held for this record. Press u to edit it again.');
        end
    end

    % Hold the QRS boundaries still for the CURRENT record.
    %
    % Every landmark is editable by default, including these. This is here only
    % for the case where you are working close to the J point and would rather
    % the QRS could not be caught by a stray drag. It lasts until you leave the
    % record and it changes nothing about the labels.
    %
    % Editing a QRS boundary always applies the change to all twelve leads, lock
    % or no lock, since the QRS is a record-level quantity under both conventions
    % and leaving eleven leads on the old value would put two answers in one
    % table. That is in setFid and is not something u turns off.
    function toggleQrsUnlock()
        S = guidata(S.fig);
        if ~TLEAD
            set(S.status,'ForegroundColor',[0.75 0.10 0.10], ...
                'String','u only applies in per-lead T mode.');
            return;
        end
        S.qrsUnlock = ~(isfield(S,'qrsUnlock') && S.qrsUnlock);
        guidata(S.fig,S);
        if S.qrsUnlock
            set(S.status,'ForegroundColor',[0.10 0.30 0.70], 'String', ...
                'QRS editable again. An edit applies to all twelve leads.');
        else
            set(S.status,'ForegroundColor',[0.75 0.10 0.10], 'String', ...
                ['QRS held for ' S.recName ' so a stray drag cannot move it. ' ...
                 'Press u to edit it again.']);
        end
        showLead();
    end

    function clearSel()
        S = guidata(S.fig);
        li = S.leadIdx; k = S.selFid;
        if k <= 0 || k > numel(S.FIDKEY); return; end
        if qrsLocked(k); return; end
        S.L(li).idx(k) = NaN;
        S.L(li).edited   = true;
        S.L(li).reviewed = true;
        if TLEAD; S.L(li).tReviewed = true; end
        S.L(li).source   = 'manual';
        S.dirty = true;
        guidata(S.fig, S);
        showLead(); renderLeadList(); autoSave();
    end

    % Bring one landmark back after it was marked absent.
    %
    % It uses the STORED seed rather than re-running the delineator. Re-running
    % it had two faults: it overwrote this lead's invertedT flag with a fresh
    % automatic guess, discarding a manual decision, and more importantly it
    % could not restore a landmark the seed never produced in the first place.
    %
    % That second fault is why "Q absent" felt dead while "R absent" worked.
    % nameQRS only names a Q when there is a negative deflection before R
    % deeper than five per cent of the complex, so on this data the seed finds
    % Q in well under half of leads and S in fewer still, while R is nearly
    % always found. Toggling an already-absent Q asked for a restore, the seed
    % returned nothing again, and the button appeared to do nothing at all.
    %
    % So when the seed has no value, the landmark is placed at the most
    % plausible spot measured from the trace itself, with NO threshold applied:
    % you have already decided the wave is there, and the threshold is exactly
    % what hid it. It is then selected, so the arrow keys or a click on the
    % trace move it straight away.
    function v = restoreValue(li, k)
        v = NaN;
        if isfield(S.L(li),'seed') && numel(S.L(li).seed) >= k
            v = S.L(li).seed(k);
        end
        if isfinite(v); return; end
        v = guessLandmark(li, k);
    end

    % Best-guess position for a landmark the automatic seed never produced.
    % Peaks are read off this lead's own trace relative to the QRS-onset
    % voltage; boundaries fall back to simple proportional placement. Every
    % result is a starting point to drag, not an answer.
    function v = guessLandmark(li, k)
        v   = NaN;
        idx = S.L(li).idx;
        n   = S.n;
        V   = double(S.tbl.(S.LEADS{li}));
        qs  = idx(1); qe = idx(5); rr = idx(3);
        if ~isfinite(qs); qs = max(1, round(0.05*n)); end
        if ~isfinite(qe); qe = min(n, qs + round(80/max(S.msps,eps))); end
        qs = min(max(1,round(qs)),n); qe = min(max(1,round(qe)),n);
        if qe <= qs; qe = min(n, qs+1); end
        base = V(qs);
        switch k
            case 3                                   % R: largest POSITIVE swing
                seg = V(qs:qe) - base;
                [~,i2] = max(seg); v = qs + i2 - 1;
            case 2                                   % Q: deepest dip before R
                hi = rr; if ~isfinite(hi); hi = round((qs+qe)/2); end
                hi = min(max(qs+1,round(hi)),qe);
                seg = V(qs:hi) - base;
                [~,i2] = min(seg); v = qs + i2 - 1;
            case 4                                   % S: deepest dip after R
                lo = rr; if ~isfinite(lo); lo = round((qs+qe)/2); end
                lo = min(max(qs,round(lo)),qe-1);
                seg = V(lo:qe) - base;
                [~,i2] = min(seg); v = lo + i2 - 1;
            case 1                                   % QRS onset
                v = max(1, qe - round(80/max(S.msps,eps)));
            case 5                                   % QRS offset
                v = min(n, qs + round(80/max(S.msps,eps)));
            case 7                                   % T peak: biggest swing after QRS
                lo = min(n, qe + round(40/max(S.msps,eps)));
                if lo >= n; lo = min(n-1, qe+1); end
                seg = abs(V(lo:n) - V(min(n,qe)));
                [~,i2] = max(seg); v = lo + i2 - 1;
            case 6                                   % T onset
                tp = idx(7); if ~isfinite(tp); tp = guessLandmark(li,7); end
                v = round(qe + 0.35*(tp - qe));
            case 8                                   % T offset
                tp = idx(7); if ~isfinite(tp); tp = guessLandmark(li,7); end
                ts = idx(6); if ~isfinite(ts); ts = round(qe + 0.35*(tp-qe)); end
                v = round(tp + (tp - ts));
        end
        if ~isfinite(v); v = round(0.5*n); end
        v = min(max(1, round(v)), n);
    end

    % Mark a whole wave absent, or restore it. The T group is the one that
    % genuinely disappears in these simulations.
    function toggleWave(grp)
        S = guidata(S.fig);
        li = S.leadIdx;
        if all(~isfinite(S.L(li).idx(grp)))
            for kk = grp
                S.L(li).idx(kk) = restoreValue(li, kk);
            end
        else
            S.L(li).idx(grp) = NaN;
        end
        S.selFid = grp(1);
        S.L(li).edited   = true;
        S.L(li).reviewed = true;
        S.L(li).source   = 'manual';
        S.dirty = true;
        guidata(S.fig,S);
        showLead(); renderLeadList(); autoSave();
    end

    % Toggle ONE landmark between absent and a value. Used for the Q, R and S
    % buttons, where the wave either exists or it does not and there is no
    % onset or offset to clear alongside it (ported from gold_label_ecg.m,
    % whose worklist supplied the restore value; here it comes from a fresh
    % auto-seed of this lead, which may itself decide the wave is absent).
    function toggleLandmark(k)
        S = guidata(S.fig);
        li = S.leadIdx;
        if k < 1 || k > numel(S.FIDKEY); return; end
        if ~isfinite(S.L(li).idx(k))
            S.L(li).idx(k) = restoreValue(li, k);   % bring the landmark back
        else
            S.L(li).idx(k) = NaN;                   % this wave does not exist
        end
        S.selFid = k;
        S.L(li).edited   = true;
        S.L(li).reviewed = true;
        S.L(li).source   = 'manual';
        S.dirty = true;
        guidata(S.fig,S);
        showLead(); renderLeadList(); autoSave();
    end

    % ---------------------- saved / unsaved status indicator
    function refreshStatus()
        S = guidata(S.fig);
        if ~isfield(S,'status') || ~isgraphics(S.status); return; end
        if S.dirty
            set(S.status, 'ForegroundColor',[0.75 0.10 0.10], 'String', 'Saving...');
        else
            dn = sum(S.revCount);
            if WLMODE && ~TLEAD
                tt = numel(S.recs);
            else
                tt = numel(S.recs)*numel(S.LEADS);
            end
            set(S.status, 'ForegroundColor',[0.10 0.50 0.10], 'String', ...
                sprintf(['Saved automatically  -  %d of %d units saved  (%.1f%%).  ' ...
                         'Edits save as you make them; a record is written when you move off it.'], ...
                        dn, tt, 100*dn/max(tt,1)));
        end
    end

    % ---------------------------------------------- mouse callbacks
    function onDown(~,~)
        S = guidata(S.fig);
        if isempty(S.tbl); return; end
        cp = get(S.ax,'CurrentPoint');
        x  = cp(1,1);
        xv = S.xview;
        if x < xv(1) || x > xv(2); return; end
        % grab the nearest marker if the click is close to one (in x)
        idx = S.L(S.leadIdx).idx; t = S.trel;
        xs = nan(1,numel(idx));
        for k = 1:numel(idx)
            if isfinite(idx(k)); xs(k) = t(min(max(1,round(idx(k))),S.n)); end
        end
        d = abs(xs - x);
        [dist, k] = min(d);
        tol = 0.03 * (xv(2)-xv(1));
        if isfinite(dist) && dist <= tol
            % A locked marker can be selected so you can read its time, but it
            % is never picked up for dragging. Letting the drag start and having
            % every move rejected downstream would look like a stuck marker.
            if qrsLocked(k)
                S = guidata(S.fig);
                S.selFid = k; guidata(S.fig,S); showLead(); return;
            end
            S.dragF  = k;
            S.selFid = k;
            guidata(S.fig,S);
            showLead();
        else
            % click away from any marker -> move the SELECTED landmark there
            [~, ix] = min(abs(t - x));
            guidata(S.fig,S);
            setFid(S.selFid, ix);
        end
    end

    function onMotion(~,~)
        S = guidata(S.fig);
        if S.dragF == 0; return; end
        cp = get(S.ax,'CurrentPoint'); x = cp(1,1);
        t = S.trel;
        x = min(max(x, t(1)), t(end));
        [~, ix] = min(abs(t - x));
        setFid(S.dragF, ix);
    end

    function onUp(~,~)
        S = guidata(S.fig);
        if S.dragF ~= 0
            S.dragF = 0; guidata(S.fig,S); renderLeadList();
            autoSave();          % one write per drag, not per mouse-move
        end
    end

    % ------------------------------------------- keyboard callback
    function onKey(~,ev)
        S = guidata(S.fig);
        if isempty(S.tbl); return; end
        ctrl = any(strcmp(ev.Modifier,'control')) || any(strcmp(ev.Modifier,'command'));
        shift= any(strcmp(ev.Modifier,'shift'));
        step = 1; if shift; step = 5; end
        cur  = S.L(S.leadIdx).idx(S.selFid);
        switch ev.Key
            case {'1','2','3','4','5','6','7','8'}
                S.selFid = min(str2double(ev.Key), numel(S.FIDS));
                guidata(S.fig,S); showLead();
            case 'leftarrow'
                if isfinite(cur); setFid(S.selFid, cur - step); end
            case 'rightarrow'
                if isfinite(cur); setFid(S.selFid, cur + step); end
            case {'delete','backspace'}
                clearSel();
            case 'o'; toggleOverlay();
            case 'm'; toggleMagnitude();
            case 'g'
                if shift; snapToffToTangent(); else; toggleTangent(); end
            case {'equal','add'};       zoomView(1/1.4);
            case {'hyphen','subtract'}; zoomView(1.4);
            case 'comma';               panView(-0.25);
            case 'period';              panView(+0.25);
            case {'f','0'};             fitView();
            case 'n'
                if ctrl; changeRecord(+1); else; changeLead(+1); end
            case 'p'
                if ctrl; changeRecord(-1); else; changeLead(-1); end
            case 'b'
                if TLEAD
                    set(S.status,'ForegroundColor',[0.75 0.10 0.10], 'String', ...
                        ['b is disabled in per-lead T mode. QRS is already ' ...
                         'shared and locked, and T is per lead by design.']);
                elseif shift
                    copyQrsToAllLeads();
                else
                    copyBoundsToAllLeads();
                end
            case 'u'; toggleQrsUnlock();
            case 'r'; reseedLead();
            case 't'; toggleWave(S.TGRP);
            case 'i'; toggleFlag('invertedT');
            case 'x'; toggleFlag('exclude');
            case 's'; saveRecord();
        end
    end

    % ---- view toggles -------------------------------------------------
    function toggleOverlay()
        S = guidata(S.fig);
        S.overlay = ~S.overlay; guidata(S.fig,S); showLead();
    end

    function toggleMagnitude()
        S = guidata(S.fig);
        S.magnitude = ~S.magnitude; guidata(S.fig,S); showLead();
    end

    function toggleTangent()
        S = guidata(S.fig);
        S.tangent = ~S.tangent; guidata(S.fig,S); showLead();
    end

    % Accept the tangent construction as the T offset (Shift+G). Saves dragging
    % the marker onto a crossing you have already decided to trust.
    function snapToffToTangent()
        S = guidata(S.fig);
        if isempty(S.tbl); return; end
        li = S.leadIdx;
        V  = double(S.tbl.(S.LEADS{li}));
        T  = tangentOffset(V, S.L(li).idx(7), S.n);
        if ~T.ok; return; end
        S.selFid = 8;
        guidata(S.fig,S);
        setFid(8, T.cross);          % setFid clamps, redraws and auto-saves
    end

    % ---- view zoom / pan (x-axis only) --------------------------------
    % These change S.xview and redraw. They never change the stored sample
    % indices, so landmarks stay put and the y-scale re-fits what is visible.
    function setView(xv)
        S = guidata(S.fig);
        t  = S.trel;
        lo = t(1); hi = t(end);
        w  = xv(2) - xv(1);
        w  = max(w, max(10*S.msps, 20));      % never zoom tighter than ~20 ms
        if w > (hi-lo); w = hi-lo; end
        c  = mean(xv);
        a  = c - w/2;  b = c + w/2;
        if a < lo; a = lo; b = lo + w; end
        if b > hi; b = hi; a = hi - w; end
        S.xview = [a, b];
        guidata(S.fig,S); showLead();
    end

    function zoomView(f)
        S = guidata(S.fig);
        xv = S.xview; c = mean(xv); w = (xv(2)-xv(1))*f;
        setView([c - w/2, c + w/2]);
    end

    function panView(frac)
        S = guidata(S.fig);
        xv = S.xview; d = (xv(2)-xv(1))*frac;
        setView([xv(1)+d, xv(2)+d]);
    end

    function fitView()
        S = guidata(S.fig);
        if isfield(S,'xfit') && ~isempty(S.xfit); S.xview = S.xfit; end
        guidata(S.fig,S); showLead();
    end

    % ---- scroll-wheel zoom, centred on the cursor ---------------------
    function onScroll(~,ev)
        S = guidata(S.fig);
        if isempty(S.tbl); return; end
        cp = get(S.ax,'CurrentPoint'); xc = cp(1,1);
        xv = S.xview;
        if xc < xv(1) || xc > xv(2); xc = mean(xv); end   % cursor off-axis -> centre
        if ev.VerticalScrollCount > 0; f = 1.4; else; f = 1/1.4; end
        r = (xc - xv(1)) / max(eps, (xv(2)-xv(1)));
        w = (xv(2)-xv(1)) * f;
        setView([xc - r*w, xc + (1-r)*w]);
    end

    % ------------------------------------------------- navigation
    function changeLead(delta)
        S = guidata(S.fig);
        if S.dirty; saveRecord(true); S = guidata(S.fig); end  % auto-save first
        S.leadIdx = min(max(1, S.leadIdx + delta), numel(S.LEADS));
        S.selFid  = 1;
        guidata(S.fig,S);
        renderLeadList(); showLead();
    end

    % Steps within the FILTERED view, so Ctrl+n / Ctrl+p walk one class when a
    % class filter is set rather than jumping out of it.
    function changeRecord(delta)
        S = guidata(S.fig);
        if numel(S.view) <= 1; return; end
        newPos = min(max(1, S.pos + delta), numel(S.view));
        if newPos == S.pos; return; end
        commitRecord();           % accept the record you are leaving
        S = guidata(S.fig);
        S.pos = newPos;
        guidata(S.fig,S);
        loadRecord();
    end

    function toggleFlag(name)
        S = guidata(S.fig);
        li = S.leadIdx;
        S.L(li).(name)   = ~S.L(li).(name);
        S.L(li).edited   = true;
        S.L(li).reviewed = true;
        S.dirty = true;
        guidata(S.fig,S);
        renderLeadList(); showLead(); autoSave();
    end

    % ---------------------------------------------------- saving
    % Auto-save ONLY when the reviewer actually changed something. S.dirty is
    % set by every manual action (moving, clearing or restoring a landmark, the
    % absent toggles, the flags, a re-seed) and cleared by a save, so it is
    % exactly the "has unsaved manual edits" test.
    %
    % It deliberately does NOT test 'seeded'. Opening a record seeds its leads
    % from the automatic delineator so something can be drawn, and the old test
    % treated that as work worth saving - which wrote a label file and marked
    % the record * the moment you looked at it. An untouched record is now left
    % with no label file at all, so * means reviewed by you.
    function autoSave()
        S = guidata(S.fig);
        if isstruct(S.L) && ~isempty(S.L) && S.dirty
            saveRecord(true);
        end
    end

    function saveRecord(quiet)
        if nargin < 1; quiet = false; end
        S = guidata(S.fig);
        % make sure every lead has at least a seed before writing
        for li = 1:numel(S.LEADS)
            if ~S.L(li).seeded; seedLead(li); S = guidata(S.fig); end
        end
        L = S.L; recName = S.recName; t = S.trel;
        meta = struct('recName',recName,'nSamples',S.n,'msPerSample',S.msps, ...
            'leads',{S.LEADS},'fids',{S.FIDKEY},'width',S.WIDTH, ...
            'activationTime',S.ACTIV,'magLeads',{S.MAGLEADS}, ...
            'savedOn',datestr(now)); %#ok<TNOW1,DATST>

        matPath = fullfile(S.outDir, [recName '_labels.mat']);
        save(matPath, 'L', 'meta');

        % Per-record .mat above is what the tool resumes from. The deliverable is
        % the single combined corrections CSV below, written in the same column
        % layout the Gold reviewer tool produces so the downstream scripts in
        % Delineation/ recognise it (see writeCorrectionsCsv).
        csvPath = S.outCsv;
        onlyLead = '';
        % In per-lead T mode every lead is its own unit, so onlyLead stays empty.
        % Setting it would also prune this record's other eleven rows from the
        % CSV on every save, which is precisely the per-lead work being done.
        if WLMODE && ~TLEAD && isKey(WL, recName); onlyLead = WL(recName); end
        srcTag = 'manual_corrected'; gateField = 'reviewed';
        if TLEAD
            srcTag = 'manual_tlead_perlead_T';
            gateField = 'tReviewed';
        end
        writeCorrectionsCsv(csvPath, L, recName, S.recClass, S.LEADS, S.FIDKEY, ...
            S.msps, S.n, S.TGRP, onlyLead, srcTag, gateField);

        % mark this record as done in the left-hand list, then mark clean
        if S.recIdx >= 1 && S.recIdx <= numel(S.revCount)
            if TLEAD
                S.revCount(S.recIdx) = sum([S.L.tReviewed]);
            elseif WLMODE && isKey(WL, S.recName)
                kk = find(strcmp(S.LEADS, WL(S.recName)), 1);
                if isempty(kk); S.revCount(S.recIdx) = 0;
                else; S.revCount(S.recIdx) = double(S.L(kk).reviewed); end
            else
                S.revCount(S.recIdx) = sum([S.L.reviewed]);
            end
        end
        S.dirty = false; guidata(S.fig, S); refreshStatus();
        renderRecList(); S = guidata(S.fig);

        if ~quiet
            fprintf('Saved %s and %s\n', matPath, csvPath);
            oldName = get(S.fig,'Name');
            set(S.fig,'Name',sprintf('Saved %s  (%s)', recName, datestr(now,'HH:MM:SS'))); %#ok<TNOW1,DATST>
            drawnow;
            pause(0.6);
            set(S.fig,'Name',oldName);
        end
    end

    function onClose(~,~)
        try; commitRecord(); catch; end
        delete(S.fig);
    end
end

% ======================================================================
%                       plain (non-nested) helpers
% ======================================================================
function [q,r,s] = nameQRS(V, qs, qe)
%NAMEQRS  Name Q, R and S inside a QRS window on the Gold convention.
%   The QRS-onset voltage is the isoelectric reference. A deflection counts
%   only if it exceeds five per cent of the largest deflection in the
%   complex, so simulator ripple is not named as a wave. R is the first
%   positive peak above that threshold, Q is the negative peak before it and
%   S the negative peak after it. A complex with no qualifying positive
%   deflection is a QS complex, so it gets a Q and neither R nor S.
    q = NaN; r = NaN; s = NaN;
    if ~all(isfinite([qs qe])); return; end
    w0 = max(1, min(qs,qe));
    w1 = min(numel(V), max(qs,qe));
    if w1 <= w0; return; end
    seg = V(w0:w1);
    dev = seg - seg(1);              % relative to the QRS-onset voltage
    amp = max(abs(dev));
    if ~isfinite(amp) || amp == 0; return; end
    % 2 percent, not 5. At 5 percent this rule found Q in only 5 of 12 leads and
    % S in 2 of 12 on real records, hiding waves that are plainly present, so the
    % reviewer had to add them by hand. A lower threshold offers them and lets the
    % reviewer remove instead, which is the easier direction to judge.
    % NOTE: Q, R and S are named HERE, not by delineate_ecg_v3, which returns no
    % QRS peaks at all. The v3_q/r/s columns therefore record this rule's output
    % and are not that delineator's error.
    thr = 0.02 * amp;

    pos = dev;  pos(pos < thr) = NaN;
    if any(isfinite(pos))
        [~, ri] = max(pos);
        r = w0 + ri - 1;
        % Q is the deepest negative deflection BEFORE R
        if ri > 1
            pre = dev(1:ri-1);
            [dq, qi] = min(pre);
            if dq < -thr; q = w0 + qi - 1; end
        end
        % S is the deepest negative deflection AFTER R
        if ri < numel(dev)
            post = dev(ri+1:end);
            [ds, si] = min(post);
            if ds < -thr; s = w0 + ri + si - 1; end
        end
    else
        % no positive deflection clears the threshold, so this is a QS
        [dq, qi] = min(dev);
        if dq < -thr; q = w0 + qi - 1; end
    end
end

function v = ivInterval(a, b)
%IVINTERVAL  a - b in ms, NaN if either end is absent.
    if isnan(a) || isnan(b); v = NaN; else; v = a - b; end
end

function bl = bmLine(name, v, lo, hi)
%BMLINE  One biomarker line, flagged with "!" when outside its usual range.
    if isnan(v)
        bl = sprintf('%-6s : --', name);
    else
        flag = ''; if v < lo || v > hi; flag = ' !'; end
        bl = sprintf('%-6s : %4.0f ms%s', name, v, flag);
    end
end

function v = fidMs(idx, k, t)
%FIDMS  Time in ms of landmark k, NaN when the landmark is absent.
    if k < 1 || k > numel(idx) || ~isfinite(idx(k)); v = NaN; return; end
    i = min(max(1,round(idx(k))), numel(t));
    v = t(i);
end

function T = tangentOffset(V, tpk, n)
%TANGENTOFFSET  T offset by the tangent method, in SAMPLES (1-based, fractional).
%   Takes the steepest limb after the T peak and extends it to the isoelectric
%   baseline; the crossing is the T offset. This is the construction the Gold
%   protocol specifies, and it is the right tool for a T wave that ends by
%   flattening, where "last return to the floor" is hard to see by eye and the
%   magnitude strip is compressed by the much larger QRS.
%
%   Returned fields: ok, cross (samples), i0 (contact sample), slope (per
%   sample), base (isoelectric level).
%
%   The baseline is the median of the last 15 percent of the RECORD, not of the
%   plotted window. These records are about 1000 ms with the beat inside the
%   first half, so the tail is clean isoelectric baseline; taking it from the
%   view would make the answer move when you zoom.
%
%   The steepest limb is found as the largest absolute derivative after the T
%   peak, so the construction works unchanged for an inverted T: there the
%   steepest limb ascends and the crossing still lands after the contact point.
    T = struct('ok',false,'cross',NaN,'i0',NaN,'slope',NaN,'base',NaN);
    if nargin < 3 || isempty(n); n = numel(V); end
    if ~isfinite(tpk); return; end
    tpk = min(max(1,round(tpk)), n);
    b0   = max(1, round(0.85*n));
    tail = V(b0:n);
    tail = tail(isfinite(tail));
    if isempty(tail); return; end
    base = median(tail);
    if ~isfinite(base); return; end
    Vs = smooth5(V);
    d  = gradient(Vs);
    lo = min(n, tpk + 1);
    if n <= lo; return; end
    seg = d(lo:n);
    [~, rel] = max(abs(seg));
    i0 = lo + rel - 1;
    sl = d(i0);
    if ~isfinite(sl) || sl == 0; return; end
    cross = i0 + (base - Vs(i0)) / sl;
    if ~isfinite(cross); return; end
    if cross < i0; return; end                 % limb points away from baseline
    cross = min(max(1, cross), n);
    T.ok = true; T.cross = cross; T.i0 = i0; T.slope = sl; T.base = base;
end

function Y = smooth5(V)
%SMOOTH5  Light 5-point moving average, so the steepest-slope search is not
%   chasing a single noisy sample. Edge-padded, so no shift is introduced.
    V = V(:).';
    k = ones(1,5)/5;
    Vp = [repmat(V(1),1,2), V, repmat(V(end),1,2)];
    Yp = conv(Vp, k, 'same');
    Y  = Yp(3:end-2);
end

function xf = beatWindow(M, LEADS, MAGLEADS, t)
%BEATWINDOW  The time window that actually contains the beat, in ms.
%   Returns [lo hi] for the default view. These simulated records run for a
%   fixed 1000 ms while the beat itself lasts only 300-600 ms, so fitting the
%   whole record left the landmarks squeezed into the left third of the axis.
%
%   The window is measured on the spatial magnitude across the independent
%   leads rather than on one lead's fiducials: it therefore answers "where is
%   there any cardiac activity at all", is identical for all twelve leads of a
%   record, and needs no delineator output. A small fraction of the peak above
%   the noise floor is used as the threshold so a flat tail is excluded but the
%   low-amplitude start and end of the beat are not.
%
%   This changes the VIEW only. Sample indices, and therefore every saved
%   fiducial, are completely unaffected.
    xf = [t(1) t(end)];
    gi = find(ismember(LEADS, MAGLEADS));
    if isempty(gi); return; end
    gi = gi(gi <= size(M,1));
    gi = gi(all(isfinite(M(gi,:)), 2));
    if isempty(gi); return; end
    D = M(gi,:);
    D = D - median(D, 2, 'omitnan');
    D(~isfinite(D)) = 0;
    mag = sqrt(sum(D.^2, 1));
    fl  = qtile(mag, 0.10);
    pk  = max(mag);
    if ~isfinite(pk) || ~isfinite(fl) || pk <= fl; return; end
    thr = fl + 0.02*(pk - fl);
    on  = find(mag > thr, 1, 'first');
    off = find(mag > thr, 1, 'last');
    if isempty(on) || isempty(off) || off <= on; return; end
    span  = t(off) - t(on);
    padms = max(0.10*span, 40);          % breathing room each side
    lo = max(t(1),   t(on)  - padms);
    hi = min(t(end), t(off) + padms);
    if hi > lo; xf = [lo hi]; end
end

function q = qtile(x, p)
%QTILE  Nearest-rank quantile of a vector, with no Statistics Toolbox
%   dependency. Used for the magnitude strip's noise floor, since prctile
%   lives in the Statistics Toolbox and this tool must run without it.
    x = x(isfinite(x));
    if isempty(x); q = NaN; return; end
    x = sort(x(:));
    n = numel(x);
    q = x(min(max(round(p*(n-1))+1, 1), n));
end

function c = classOf(rec)
%CLASSOF  The class label used by the filter. A record with no class of its own
%   (the single-table example files carry none) is grouped as 'unlabelled'
%   rather than being given an invented disease class.
    c = '';
    if isfield(rec,'class'); c = char(rec.class); end
    if isempty(c); c = 'unlabelled'; end
end

function cs = uniqueClasses(recs)
%UNIQUECLASSES  Sorted distinct class labels across all records, for the popup.
%   Built with an explicit loop over a cellstr. An earlier arrayfun version
%   concatenated the names into one long char row ('HealthyAnterior...'), so
%   unique() returned unique CHARACTERS and the popup offered no real classes.
    cs = {};
    if isempty(recs); return; end
    names = cell(1, numel(recs));
    for i = 1:numel(recs)
        names{i} = classOf(recs(i));
    end
    cs = unique(names);      % unique() on a cellstr keeps whole names
    cs = cs(:).';            % row, so [{'all'}, cs] builds a flat popup list
end

function recs = buildRecords(files)
%BUILDRECORDS  Expand the input .mat file(s) into a flat list of reviewable
%   records. Each record is a struct with fields:
%       name   display / label-file name (filesystem-safe)
%       class  disease class if known (e.g. 'HealthyECGs' -> 'Healthy'), else ''
%       tbl    the ECG table held in memory (multi-record files), or []
%       file   the .mat path to load on demand (plain single-record files)
%   A plain file that holds one table / one ECG struct gives a single record
%   loaded lazily by path. A multi-record file whose variables are cell arrays
%   (or struct arrays) of ECG tables - the SimulatedECGs_Smith2026.mat layout,
%   with one cell array per disease class - gives one in-memory record per
%   table, tagged with the class taken from the variable name.
    recs = struct('name',{},'class',{},'tbl',{},'file',{});
    for fi = 1:numel(files)
        f = files{fi};
        [~, base] = fileparts(f);
        raw = load(f);
        added = expandRaw(raw, base);      % multi-record containers, if any
        if isempty(added)
            recs(end+1) = struct('name',base,'class','','tbl',[],'file',f); %#ok<AGROW>
        else
            recs = [recs, added]; %#ok<AGROW>
        end
    end
    if isempty(recs)
        error('manual_label_ecg:norecords', ...
            'No ECG records were found in the selected input.');
    end
end

function recs = expandRaw(raw, base)
%EXPANDRAW  Find multi-record containers in a loaded .mat and return one record
%   per ECG table inside them. A container is a cell array of tables/structs or
%   a non-scalar struct array; each element that looks like a 12-lead ECG
%   becomes its own record. Scalar structs and lone table variables are NOT
%   expanded here (pickTable handles those as a single record), so ecg_table.mat
%   and all_1_table.mat keep loading exactly as before. Returns an empty struct
%   array when nothing expandable is present.
    recs = struct('name',{},'class',{},'tbl',{},'file',{});
    fn = fieldnames(raw);
    for i = 1:numel(fn)
        v = raw.(fn{i});
        cls = stripECGs(fn{i});
        items = {};
        if iscell(v)
            items = v(:).';
        elseif isstruct(v) && numel(v) > 1
            items = arrayfun(@(e) e, v(:).', 'UniformOutput', false);
        end
        for k = 1:numel(items)
            t = toEcgTable(items{k});
            if ~isempty(t)
                nm = sprintf('%s_%03d', cls, k);
                recs(end+1) = struct('name',nm,'class',cls,'tbl',t,'file',''); %#ok<AGROW>
            end
        end
    end
end

function t = toEcgTable(x)
%TOECGTABLE  Coerce one container element to a 12-lead ECG table, or [] if it
%   does not look like an ECG. A table is accepted when its variable names
%   resemble a 12-lead record; a scalar struct is rebuilt with the shared
%   struct->table helper.
    t = [];
    if istable(x)
        if looksLikeEcg(x.Properties.VariableNames); t = x; end
    elseif isstruct(x) && isscalar(x) && looksLikeEcg(fieldnames(x))
        t = buildTableFromStruct(x);
    end
end

function c = stripECGs(name)
%STRIPECGS  Turn a container variable name into a short class label, dropping a
%   trailing 'ECGs'/'ECG' (e.g. 'AnteriorInfarctionECGs' -> 'AnteriorInfarction').
    c = char(name);
    c = regexprep(c, 'ECGs?$', '');
    if isempty(c); c = char(name); end
end

function tbl = pickTable(raw)
%PICKTABLE  Return an ECG record as a MATLAB table from a loaded .mat.
%   Accepts three storage styles:
%     1) a variable that is already a table (e.g. all_1_table.mat),
%     2) a scalar struct whose fields are the columns (e.g. ecg_table.mat,
%        which stores a struct with fields Time, I, ... V6),
%     3) the loaded file itself holding the columns as separate variables.
    tbl = [];
    fn = fieldnames(raw);

    % 1) a genuine table variable
    for i = 1:numel(fn)
        v = raw.(fn{i});
        if istable(v); tbl = v; return; end
    end

    % 2) a scalar struct variable that looks like an ECG record
    for i = 1:numel(fn)
        v = raw.(fn{i});
        if isstruct(v) && isscalar(v) && looksLikeEcg(fieldnames(v))
            tbl = buildTableFromStruct(v);
            if ~isempty(tbl); return; end
        end
    end

    % 3) the loaded workspace itself carries the columns
    if looksLikeEcg(fn)
        tbl = buildTableFromStruct(raw);
    end
end

function tf = looksLikeEcg(names)
%LOOKSLIKEECG  True if a set of field names resembles a 12-lead ECG record.
    leads = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
    nLead = sum(ismember(leads, names));
    hasTime = any(ismember({'Time','t'}, names));
    tf = (nLead >= 6) && (hasTime || nLead >= 8);
end

function tbl = buildTableFromStruct(s)
%BUILDTABLEFROMSTRUCT  Assemble a table from the numeric vector fields of a
%   struct, keeping only recognised time/lead columns of equal length.
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
                    keep{end+1} = nm;   %#ok<AGROW>
                    cols{end+1} = v;    %#ok<AGROW>
                end
            end
        end
    end
    if numel(keep) < 2
        tbl = [];
    else
        tbl = table(cols{:}, 'VariableNames', keep);
    end
end

function L = emptyLabels(leads, nFid)
%EMPTYLABELS  One struct entry per lead, un-seeded.
    n = numel(leads);
    % tReviewed is the per-lead T pass's own flag. It exists separately from
    % reviewed because the label files written before worklist mode carry
    % reviewed = true on all twelve leads, so reusing that flag would make the
    % per-lead pass believe every lead was already done on the first launch.
    tmpl = struct('lead','','idx',nan(1,nFid),'invertedT',false, ...
        'exclude',false,'seeded',false,'edited',false,'reviewed',false, ...
        'tReviewed',false,'seed',nan(1,nFid),'source','');
    L = repmat(tmpl,1,n);
    for i = 1:n; L(i).lead = leads{i}; end
end

function Lnew = migrateLabels(Lold, Lnew, fidkey)
%MIGRATELABELS  Carry an older six-point label set into the current eight
%   slots, matching on the landmark names that both sets share. Anything the
%   old set did not hold stays absent and is filled by the next re-seed.
    oldkey = {'QRS_start','R_peak','QRS_end','T_start','T_peak','T_end'};
    for i = 1:min(numel(Lold), numel(Lnew))
        if ~isfield(Lold(i),'idx'); continue; end
        oi = Lold(i).idx;
        if numel(oi) ~= numel(oldkey); continue; end
        for k = 1:numel(oldkey)
            j = find(strcmp(fidkey, oldkey{k}), 1);
            if ~isempty(j); Lnew(i).idx(j) = oi(k); end
        end
        if isfield(Lold(i),'invertedT'); Lnew(i).invertedT = Lold(i).invertedT; end
        if isfield(Lold(i),'exclude');   Lnew(i).exclude   = Lold(i).exclude;   end
        if isfield(Lold(i),'edited');    Lnew(i).edited    = Lold(i).edited;    end
        if isfield(Lold(i),'source');    Lnew(i).source    = Lold(i).source;    end
        Lnew(i).seeded = true;
    end
end

function idx = defaultIdx(n)
%DEFAULTIDX  Sensible fallback spread across a beat of length n, for the six
%   landmarks the automatic seed supplies.
%   Order: QRS start, R peak, QRS end, T start, T peak, T end.
    idx = round([0.06 0.10 0.14 0.30 0.42 0.55]*n);
    idx = max(1, min(n, idx));
end

function idx = clampIdx(idx, n)
%CLAMPIDX  Round and clamp to 1..n, leaving an absent landmark as NaN.
    ok = isfinite(idx);
    idx(ok) = max(1, min(n, round(idx(ok))));
    idx(~ok) = NaN;
end

function writeCorrectionsCsv(outCsv, L, recName, recClass, leads, fidkey, msps, n, TGRP, onlyLead, srcTag, gateField)
%WRITECORRECTIONSCSV  Append/replace this record's 12 lead rows in ONE combined
%   corrections CSV, in the column layout that Gold/tool/gold_label_ecg.m writes
%   (Gold/corrections/gold_worklist_*_corrections.csv). Using the same column
%   names means the existing Delineation scripts read these rows by name without
%   being taught a new schema.
%
%   Columns (Gold's 29, then two internal extras):
%     record_id disease_class lead beat_id fs_hz n_samples
%     p_onset_sample p_peak_sample p_offset_sample
%     qrs_onset_sample q_peak_sample r_peak_sample s_peak_sample qrs_offset_sample
%     t_onset_sample t_peak_sample t_offset_sample
%     p_present qrs_present q_present r_present s_present t_present
%     flags also_delineator priority label_source reviewed edited_at
%     invertedT exclude
%     v3_*_sample   the delineate_ecg_v3 seed, exactly as it was before you
%                   corrected it, so (landmark - v3_landmark) is that
%                   delineator's signed error on a human-checked unit
%
%   The three P columns are written EMPTY with p_present = 0: this simulated data
%   contains no P wave (see README), so a blank is the honest value rather than a
%   fabricated sample. invertedT and exclude are appended at the end because they
%   are per-lead judgements this tool records and Gold's schema has no slot for;
%   a name-based reader ignores them.
%
%   Landmark name mapping, internal -> Gold/pipeline:
%     QRS_start -> qrs_onset_sample     T_start -> t_onset_sample
%     Q_peak    -> q_peak_sample        T_peak  -> t_peak_sample
%     R_peak    -> r_peak_sample        T_end   -> t_offset_sample
%     S_peak    -> s_peak_sample
%     QRS_end   -> qrs_offset_sample
%
%   Sample indices are 0-based on write (the pipeline's convention) whereas the
%   tool holds 1-based MATLAB indices, so 1 is subtracted.
%
%   Only leads you explicitly marked reviewed (the r key) are written. An
%   unchecked lead still carries the automatic seed, so exporting it as
%   manual_corrected would put delineator output into a human-truth test set.
%
%   The file is merged, not overwritten: only rows for (record_id, lead) pairs of
%   THIS record are replaced, every other row on disk is preserved verbatim, and
%   the result is written through a temp file so an interrupted save cannot
%   truncate the CSV. A lead that was reviewed earlier and is not in this pass
%   keeps its row.
    if nargin < 10; onlyLead = ''; end
    % label_source names WHICH pass produced the row, so a downstream script can
    % tell a per-lead T placement from a record-level one without guessing.
    if nargin < 11 || isempty(srcTag); srcTag = 'manual_corrected'; end
    % Which per-lead flag decides that a unit may be exported. The per-lead T
    % pass gates on its own flag, since the label files it inherits carry
    % reviewed = true on leads no reviewer ever opened.
    if nargin < 12 || isempty(gateField); gateField = 'reviewed'; end
    cols = {'record_id','disease_class','lead','beat_id','fs_hz','n_samples', ...
            'p_onset_sample','p_peak_sample','p_offset_sample', ...
            'qrs_onset_sample','q_peak_sample','r_peak_sample','s_peak_sample', ...
            'qrs_offset_sample','t_onset_sample','t_peak_sample','t_offset_sample', ...
            'p_present','qrs_present','q_present','r_present','s_present','t_present', ...
            'flags','also_delineator','priority','label_source','reviewed', ...
            'edited_at','invertedT','exclude', ...
            'v3_qrs_onset_sample','v3_q_peak_sample','v3_r_peak_sample', ...
            'v3_s_peak_sample','v3_qrs_offset_sample','v3_t_onset_sample', ...
            'v3_t_peak_sample','v3_t_offset_sample'};
    header = strjoin(cols, ',');

    fs = 1000/msps;
    if ~isfinite(fs) || fs <= 0; fs = 1000; end
    stamp = datestr(now,'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>

    kq = find(strcmp(fidkey,'Q_peak'),1);
    kr = find(strcmp(fidkey,'R_peak'),1);
    ks = find(strcmp(fidkey,'S_peak'),1);

    % ---- build this record's rows -------------------------------------
    fresh = containers.Map('KeyType','char','ValueType','char');
    order = {};
    for i = 1:numel(leads)
        % ONLY export units you explicitly checked. An unreviewed lead still
        % holds raw delineate_ecg_v3 output; writing it as manual_corrected /
        % reviewed = 1 would put machine labels into a human-truth test set,
        % which is the one error that would silently invalidate the evaluation.
        if ~isempty(onlyLead) && ~strcmp(leads{i}, onlyLead); continue; end
        if ~(isfield(L(i),gateField) && L(i).(gateField)); continue; end
        id = L(i).idx;
        g  = @(name) fidStr(id, fidkey, name);
        if isfield(L(i),'seed'); sdv = L(i).seed; else; sdv = nan(size(id)); end
        sd = @(name) fidStr(sdv, fidkey, name);
        qp = double(isfinite(id(kq)));
        rp = double(isfinite(id(kr)));
        sp = double(isfinite(id(ks)));
        tp = double(any(isfinite(id(TGRP))));
        % QRS is present if any of its five landmarks survived review
        qrsIdx = find(ismember(fidkey, {'QRS_start','Q_peak','R_peak','S_peak','QRS_end'}));
        qrsp = double(any(isfinite(id(qrsIdx))));
        flags = 'SMITH2026 sim | no P wave';
        if L(i).invertedT; flags = [flags ' | inverted T']; end
        if L(i).exclude;   flags = [flags ' | EXCLUDED']; end
        vals = { recName, recClass, leads{i}, '1', num2str(fs), num2str(n), ...
                 '', '', '', ...
                 g('QRS_start'), g('Q_peak'), g('R_peak'), g('S_peak'), g('QRS_end'), ...
                 g('T_start'), g('T_peak'), g('T_end'), ...
                 '0', num2str(qrsp), num2str(qp), num2str(rp), num2str(sp), num2str(tp), ...
                 flags, '0', '1', srcTag, '1', ...
                 stamp, num2str(double(L(i).invertedT)), num2str(double(L(i).exclude)), ...
                 sd('QRS_start'), sd('Q_peak'), sd('R_peak'), sd('S_peak'), ...
                 sd('QRS_end'), sd('T_start'), sd('T_peak'), sd('T_end') };
        key = [recName '|' leads{i}];
        fresh(key) = strjoin(vals, ',');
        order{end+1} = key; %#ok<AGROW>
    end

    % ---- merge with whatever is already on disk ------------------------
    kept = {}; seen = containers.Map('KeyType','char','ValueType','logical');
    if exist(outCsv,'file') == 2
        try; txt = fileread(outCsv); catch; txt = ''; end
        lines = regexp(txt, '\r\n|\n|\r', 'split');
        for li = 1:numel(lines)
            ln = lines{li};
            if isempty(strtrim(ln)); continue; end
            if startsWith(ln, 'record_id,'); continue; end
            parts = strsplit(ln, ',', 'CollapseDelimiters', false);
            if numel(parts) < 3; continue; end
            k = [parts{1} '|' parts{3}];     % record_id | lead
            if isKey(seen,k); continue; end
            seen(k) = true;
            % Prune rows this record left behind under the all-leads scheme:
            % in worklist mode only its assigned lead belongs in the file.
            if ~isempty(onlyLead) && strcmp(parts{1}, recName) ...
                    && ~strcmp(parts{3}, onlyLead)
                continue;
            end
            if isKey(fresh,k)
                kept{end+1} = fresh(k);      % replaced by this session's version
            else
                kept{end+1} = ln;            % another record, preserved verbatim
            end
        end
    end
    for a = 1:numel(order)
        if ~isKey(seen, order{a})
            kept{end+1} = fresh(order{a}); seen(order{a}) = true; %#ok<AGROW>
        end
    end

    % ---- atomic write --------------------------------------------------
    tmp = [outCsv '.tmp'];
    fid = fopen(tmp,'w');
    if fid < 0; warning('Cannot write %s', tmp); return; end
    fprintf(fid,'%s\n', header);
    for a = 1:numel(kept); fprintf(fid,'%s\n', kept{a}); end
    fclose(fid);
    ok = true;
    try; movefile(tmp, outCsv, 'f'); catch; ok = false; end
    if ~ok; warning('Atomic replace failed; edits are in %s', tmp); end
end

function ids = readIdList(csvPath)
%READIDLIST  The record_id column of a small CSV, as a cell array of char.
%   Used for the subset file. Read by column name so a second column can be
%   added to the file without breaking this.
    ids = {};
    if exist(csvPath,'file') ~= 2; return; end
    try; txt = fileread(csvPath); catch; return; end
    lines = regexp(txt, '\r\n|\n|\r', 'split');
    if numel(lines) < 2; return; end
    hdr = strtrim(strsplit(lines{1}, ',', 'CollapseDelimiters', false));
    c   = find(strcmp(hdr,'record_id'), 1);
    if isempty(c); return; end
    for i = 2:numel(lines)
        if isempty(strtrim(lines{i})); continue; end
        p = strsplit(lines{i}, ',', 'CollapseDelimiters', false);
        if numel(p) < c; continue; end
        v = strtrim(p{c});
        if ~isempty(v); ids{end+1} = v; end %#ok<AGROW>
    end
end

function [B, P] = readRecordBounds(csvPath)
%READRECORDBOUNDS  Two maps out of the combined corrections CSV.
%
%   B   record_id      -> [qrs_onset qrs_offset], 0-based
%   P   record_id|lead -> [q_peak r_peak s_peak t_peak], 0-based, NaN where the
%                         reviewer marked the wave absent
%
%   P exists so the per-lead T pass can put back the peaks a reviewer placed by
%   hand instead of re-deriving them. Re-deriving would silently discard the one
%   part of the first pass that was genuinely hand work, since the boundaries in
%   that file came from apply_boundary_rule and the peaks did not.
%
%   Read by column NAME rather than by position, so a column added to the file
%   later cannot shift the answer. The first row found for a record wins, which
%   is correct under both schemes: the worklist pass writes one row per record,
%   and the per-lead T pass writes twelve rows whose QRS values are identical by
%   construction and are checked as such in the readout.
%
%   A row missing either boundary is skipped rather than defaulted. A record
%   with no usable row simply has no entry, and the caller decides what that
%   means, which for the tool is that the record has no agreed QRS to inherit.
%
%   readtable is deliberately not used. The file is written by this same tool
%   through a plain fprintf, blank cells are meaningful, and a text-mode parse
%   preserves them where readtable's type inference does not.
    B = containers.Map('KeyType','char','ValueType','any');
    P = containers.Map('KeyType','char','ValueType','any');
    if exist(csvPath,'file') ~= 2; return; end
    try; txt = fileread(csvPath); catch; return; end
    lines = regexp(txt, '\r\n|\n|\r', 'split');
    if numel(lines) < 2; return; end
    hdr  = strtrim(strsplit(lines{1}, ',', 'CollapseDelimiters', false));
    col  = @(nm) find(strcmp(hdr, nm), 1);
    cRec = col('record_id'); cLead = col('lead');
    cOn  = col('qrs_onset_sample'); cOff = col('qrs_offset_sample');
    cQ = col('q_peak_sample'); cR = col('r_peak_sample');
    cS = col('s_peak_sample'); cT = col('t_peak_sample');
    if isempty(cRec) || isempty(cOn) || isempty(cOff); return; end
    num = @(parts, c) localNum(parts, c);
    for i = 2:numel(lines)
        ln = lines{i};
        if isempty(strtrim(ln)); continue; end
        pr = strsplit(ln, ',', 'CollapseDelimiters', false);
        if numel(pr) < max([cRec cOn cOff]); continue; end
        rid = strtrim(pr{cRec});
        if isempty(rid); continue; end
        a = num(pr, cOn); b = num(pr, cOff);
        if isfinite(a) && isfinite(b) && ~isKey(B, rid); B(rid) = [a b]; end
        if ~isempty(cLead) && numel(pr) >= cLead
            key = [rid '|' strtrim(pr{cLead})];
            if ~isKey(P, key)
                P(key) = [num(pr,cQ) num(pr,cR) num(pr,cS) num(pr,cT)];
            end
        end
    end
end

function v = localNum(parts, c)
%LOCALNUM  One CSV cell as a number, NaN when blank or out of range.
    v = NaN;
    if isempty(c) || numel(parts) < c; return; end
    t = strtrim(parts{c});
    if isempty(t); return; end
    v = str2double(t);
end

function s = fidStr(idx, fidkey, name)
%FIDSTR  One landmark as a 0-based sample string, empty when absent.
%   Empty means "reviewed and genuinely not there", matching the pipeline's
%   convention of a blank landmark paired with its *_present = 0.
    s = '';
    k = find(strcmp(fidkey, name), 1);
    if isempty(k); return; end
    v = idx(k);
    if isfinite(v); s = num2str(round(v) - 1); end
end
