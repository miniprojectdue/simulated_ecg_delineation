function gold_label_ecg(batchCsv, datasetRoot, outCsv)
%GOLD_LABEL_ECG  Gold-standard fiducial reviewer for MedalCare-XL.
%   A patched copy of manual_labelling/tool/medalcare_label_ecg.m. The
%   original is unchanged. The functional differences are the 12-lead
%   overlay and the spatial magnitude strip described below.
%
%   -- 12-lead overlay ---------------------------------------------------
%   All twelve leads of the current record are drawn behind the reference
%   lead in grey, amplitude-normalised to the reference lead over the
%   visible window so they can be compared on shape instead of on scale.
%   A wave boundary is a property of the whole heartbeat and not of one
%   lead, so the true onset is the earliest point at which ANY lead leaves
%   baseline and the true offset is the latest point at which ANY lead
%   returns to it. Reviewing one lead alone cannot see that, which is the
%   bias this folder exists to remove. Press o to toggle the overlay.
%
%   -- Spatial magnitude strip -------------------------------------------
%   The orange curve below the trace is the root sum of squares across the
%   eight INDEPENDENT leads, I, II and V1 to V6, each with its own baseline
%   removed first. III, aVR, aVL and aVF are exact linear combinations of
%   I and II and are left out, since including them would count the frontal
%   plane three times over.
%
%   This is the single curve the overlay can only approximate. Squaring
%   removes polarity, so an inverted lead such as aVR contributes exactly
%   as much as an upright one. Summing over eight axes removes the
%   perpendicularity problem, since a vector lying perpendicular to one
%   lead is not perpendicular to all eight. The result rests on a noise
%   floor and lifts the instant any part of the heart depolarises.
%
%   The dotted orange line is that noise floor, taken as the 10th centile
%   of the magnitude over the beat window. Read an ONSET as the first
%   sustained lift off the floor and an OFFSET as the last return to it.
%   Every fiducial line spans the strip, so you can see directly whether a
%   landmark sits at lift-off or some milliseconds after it. Press m to
%   toggle the strip.
%
%   -- Spatial velocity strip --------------------------------------------
%   Press v for a second strip, drawn in teal below the magnitude one. It
%   is the root sum of squares of the first derivative of the same eight
%   leads. A derivative carries no baseline term at all, so this curve is
%   blind to any residual offset and rises only where the signal is
%   actually moving rather than merely displaced.
%
%   Measured on this dataset its QRS foot is about half as long as the
%   magnitude foot, 9 ms against 19 ms between the same two heights, which
%   makes the QRS onset and offset easier to fix by eye. It is weaker at
%   the T offset, where a T wave ends by flattening and a derivative
%   therefore has nothing left to report, and its P foot sits well inside
%   the P wave. Use it for the QRS boundaries and keep reading the P and T
%   boundaries off the magnitude curve. It is off by default.
%

%
%   gold_label_ecg
%       No default worklist: a file dialog opens (in manual_labelling/data/) so
%       you choose which CSV to load, e.g. final_data_units.csv (clean-only,
%       194,680 units) or all_units_worklist.csv. Use the class and qc_status
%       filters to work through the units.
%   gold_label_ecg(WORKLIST_CSV)
%       WORKLIST_CSV is the worklist file to load, e.g.
%       'manual_labelling/data/final_data_units.csv'. The dataset root and
%       output folder are inferred from the repo layout.
%   gold_label_ecg(BATCH_CSV, DATASET_ROOT)
%       DATASET_ROOT is the folder that contains WP2_largeDataset_Noise/
%       (i.e. the repository root). Used to resolve each row's path_raw.
%   gold_label_ecg(BATCH_CSV, DATASET_ROOT, OUT_CSV)
%       OUT_CSV is where corrections are (auto-)saved. Default
%       <repo>/manual_labelling/data/corrections/<batch>_corrections.csv
%
%   Each worklist row is one (record, lead) unit, already reduced to ONE
%   representative interior beat with its 11 ECGdeli
%   fiducials. You correct that beat, merge_manual_corrections.py later
%   propagates the fix to every beat of the unit by R-peak alignment.
%   Landmarks (canonical order)
%       P_on  P_pk  P_off | QRS_on  Q  R  S  QRS_off | T_on  T_pk  T_off
%
%   The disease class is shown for every unit because a "wrong-looking"
%   interval is often GENUINE for the disease (long PR in AV block, wide
%   QRS in LBBB/RBBB, inverted-T / long-QT in MI) and must NOT be changed.
%   A class-specific reminder is shown under the trace.
%
%   -- Controls ----------------------------------------------------------
%     * Drag a fiducial line to move it (snaps to 1 sample = 2 ms).
%     * Click a fiducial in the right-hand list (or its line) to select it,
%       then LEFT/RIGHT arrows nudge it (Shift = 5 samples), or click on the
%       trace to jump the selected landmark there (this also RESTORES an
%       absent landmark).
%     * "P absent" / "T absent" clear (or restore) that whole wave and set
%       its presence flag, "Clear sel" marks the selected landmark absent.
%     * "Q absent" / "R absent" / "S absent" do the same for one landmark
%       each. Q, R and S have no onset or offset of their own, so presence
%       is the whole statement about them. A QS complex has no R and no S,
%       an rS complex has no Q, and a monophasic R has neither Q nor S.
%       Press the button again to restore the landmark from the worklist.
%     * n / p  next / prev unit,  r  mark reviewed & next,  s  save now.
%     * o  toggle the 12-lead overlay on / off.
%     * r  is REQUIRED for the gold protocol. Press it on EVERY record, even
%       one you inspect and correctly leave unchanged. A record that is never
%       marked is absent from the corrections file, and an offset estimated
%       only from records that needed changing is too large.
%
%   -- Output ------------------------------------------------------------
%   A corrections CSV with exactly the columns merge_manual_corrections.py
%   reads record_id, disease_class, lead, beat_id, fs_hz, n_samples, the 11
%   *_sample values (empty = absent), p/qrs/q/r/s/t_present, flags,
%   also_delineator, priority, label_source=manual_corrected, reviewed (0/1),
%   edited_at.
%   It auto-saves when you switch unit, mark reviewed, or close the window.
%
%   Requires MATLAB with a desktop figure (R2019b+ recommended). 
%   Also note that the current code has not been optimized or standardized to work with all types of ECGs.
%.  Will need optimization and standardization to work with other formats. 

LEADS  = {'I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'};
FKEYS  = {'p_onset_sample','p_peak_sample','p_offset_sample', ...
          'qrs_onset_sample','q_peak_sample','r_peak_sample','s_peak_sample', ...
          'qrs_offset_sample','t_onset_sample','t_peak_sample','t_offset_sample'};
FLABEL = {'P on','P pk','P off','QRS on','Q','R','S','QRS off','T on','T pk','T off'};
FCOL   = [0.00 0.45 0.74;   % P on    blue
          0.30 0.75 0.93;   % P pk    sky
          0.00 0.60 0.55;   % P off   teal
          0.20 0.70 0.20;   % QRS on  green
          0.60 0.65 0.00;   % Q       olive
          0.00 0.00 0.00;   % R       black
          0.95 0.55 0.00;   % S       orange
          0.85 0.15 0.15;   % QRS off red
          0.90 0.20 0.65;   % T on    magenta
          0.55 0.15 0.85;   % T pk    purple
          0.50 0.32 0.12];  % T off   brown
PGRP = 1:3; QGRP = 4:8; TGRP = 9:11;          % landmark groups
QPK  = 5;   RPK  = 6;   SPK  = 7;             % the three QRS peaks, each alone
PAD  = 60;                                     % samples of context each side

CLASSHINT = containers.Map( ...
  {'sinus','avblock','iab','lae','fam','rbbb','lbbb','mi'}, { ...
   'Normal — usually only a low-amplitude-lead P needs fixing.', ...
   'Long PR (>200 ms) is GENUINE — keep it and fix only a mis-located P.', ...
   'Wide/notched/biphasic P is the disease signature — span the whole P, do not clip it.', ...
   'Wide/biphasic P (atrial enlargement) is genuine — span the whole P.', ...
   'Low-amplitude/fractionated P — decide P present vs absent per lead.', ...
   'Wide QRS (~120-160 ms) is GENUINE — place onset/J-point on the wide complex and do not narrow it.', ...
   'Wide QRS (~120-160 ms) is GENUINE — place onset/J-point on the wide complex and do not narrow it.', ...
   'Inverted/biphasic T and long QT are GENUINE — only move T-offset if it sits on the wrong feature.'});

% -------------------------------------------------------- resolve inputs
if nargin < 1 || isempty(batchCsv)
    % No worklist given -> there is NO default. the user must choose which CSV.
    % Either pass a path explicitly, e.g.
    %   gold_label_ecg('Gold/data/gold_worklist_calibration.csv')
    %   gold_label_ecg('Gold/data/gold_worklist_test.csv')
    % or pick one in the dialog that opens (it starts in the data/ folder).
    here    = fileparts(mfilename('fullpath'));
    dataDir = fullfile(here, '..', 'data');
    startIn = fullfile(dataDir, '*.csv');
    if exist(dataDir,'dir') ~= 7; startIn = '*.csv'; end
    [fn,fp] = uigetfile(startIn, 'Select the gold worklist CSV (calibration or test)');
    if isequal(fn,0); disp('Cancelled - no worklist selected.'); return; end
    batchCsv = fullfile(fp,fn);
end
fprintf('Worklist: %s\n', batchCsv);   % echo which CSV is actually loaded
if exist(batchCsv,'file') ~= 2
    error('gold:batch','Cannot find worklist "%s".', batchCsv);
end

if nargin < 2 || isempty(datasetRoot)
    datasetRoot = inferRepoRoot(batchCsv);
    if isempty(datasetRoot)
        datasetRoot = uigetdir(pwd, ...
            'Select the repository root (folder containing WP2_largeDataset_Noise)');
        if isequal(datasetRoot,0); disp('Cancelled.'); return; end
    end
end

if nargin < 3 || isempty(outCsv)
    [bp, bn] = fileparts(batchCsv);
    corrDir = fullfile(bp, '..', 'corrections');
    if exist(corrDir,'dir') ~= 7
        try mkdir(corrDir); catch; corrDir = bp; end
    end
    outCsv = fullfile(corrDir, [bn '_corrections.csv']);
end

% -------------------------------------------------------------- load worklist
T = readtable(batchCsv, 'TextType','string');
req = [{'record_id','disease_class','lead','beat_id','fs_hz','n_samples'}, ...
       FKEYS, {'p_present','qrs_present','t_present','beat_start_sample', ...
       'beat_end_sample','flags','also_delineator','priority','path_raw'}];
miss = req(~ismember(req, T.Properties.VariableNames));
if ~isempty(miss)
    error('medalcare:cols','Worklist is missing columns: %s', strjoin(miss, ', '));
end
nUnits = height(T);
hasStatus = ismember('qc_status', T.Properties.VariableNames);   % all-units worklist?
LISTCAP   = 20000;  % max rows drawn in the left list (a full class+qc combo fits
                    % only huge views like all/all or all-clean exceed it). The list
                    % is rebuilt only on filter/review change, so this stays fast.

% -------------------------------------------------------------- state
S = struct();
S.T        = T;
S.batchCsv = batchCsv;
S.root     = datasetRoot;
S.outCsv   = outCsv;
S.LEADS    = LEADS;
S.FKEYS    = FKEYS;
S.FLABEL   = FLABEL;
S.FCOL     = FCOL;
S.PGRP     = PGRP;  S.QGRP = QGRP;  S.TGRP = TGRP;
S.QPK      = QPK;   S.RPK  = RPK;   S.SPK  = SPK;
S.PAD      = PAD;
S.HINT     = CLASSHINT;
S.classFilter  = 'all';
S.statusFilter = 'all';
S.hasStatus    = hasStatus;
S.LISTCAP      = LISTCAP;
S.view     = 1:nUnits;      % row indices currently listed
S.pos      = 1;             % position within view
S.sel      = 0;             % selected fiducial index (0 = none)
S.dragF    = 0;             % fiducial being dragged
S.fv       = nan(1,numel(FKEYS));   % working fiducial samples for current unit
S.win      = [0 1];
S.xview    = [];                    % visible x-limits in ms (zoom/pan); set per unit
S.xfit     = [];                    % remembered auto-fit window ('f' restores it)
S.xslBusy  = false;                 % guard: slider being set programmatically
S.overlay  = true;                  % 12-lead overlay on by default ('o' toggles)
S.magnitude = true;                 % spatial magnitude strip on by default ('m' toggles)
% The eight INDEPENDENT leads. III, aVR, aVL and aVF are exact linear combinations
% of I and II, so including them would triple-count the frontal plane and tilt the
% magnitude toward frontal activity that the precordial leads see only once.
S.MAGLEADS = {'I','II','V1','V2','V3','V4','V5','V6'};
% The velocity strip is off by default. It is a QRS aid rather than a replacement,
% and leaving it off keeps the default view identical to the one the earlier records
% were reviewed under. Press v when working a QRS boundary.
S.velocity = false;
S.sig      = containers.Map('KeyType','char','ValueType','any'); % record -> 12xN
S.corr     = containers.Map('KeyType','char','ValueType','any'); % key -> struct
S.dirty    = false;

% resume load any existing corrections file
if exist(outCsv,'file') == 2
    S.corr = readCorrections(outCsv, FKEYS);
end

% deleted records (whole-record exclusions) auto-saved next to the corrections
% file and applied later by apply_deletions.py. Raw signal files are kept.
[corrDir0, ~, ~] = fileparts(outCsv);
S.delCsv  = fullfile(corrDir0, 'deleted_records.csv');
S.deleted = readDeletions(S.delCsv);
if ~isempty(keys(S.deleted))
    S.view = S.view(~ismember(S.T.record_id(S.view), string(keys(S.deleted))));
end

% -------------------------------------------------------------- GUI
S.fig = figure('Name','MedalCare-XL fiducial corrector','NumberTitle','off', ...
    'Color','w','Units','normalized','Position',[0.05 0.08 0.9 0.84], ...
    'WindowButtonDownFcn',@onDown,'WindowButtonMotionFcn',@onMotion, ...
    'WindowButtonUpFcn',@onUp,'KeyPressFcn',@onKey,'CloseRequestFcn',@onClose, ...
    'WindowScrollWheelFcn',@onScroll);

% unit list (left)
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.015 0.955 0.20 0.03],'String','Units  (filter: class / qc)', ...
    'HorizontalAlignment','left','BackgroundColor','w','FontWeight','bold');
S.classPop = uicontrol('Style','popupmenu','Parent',S.fig,'Units','normalized', ...
    'Position',[0.015 0.915 0.098 0.03],'String',[{'all'}, uniqueClasses(T)], ...
    'TooltipString','disease class','Callback',@(h,~)onFilter());
if hasStatus
    statusOpts = [{'all'}, uniqueStatus(T)];
else
    statusOpts = {'all'};
end
S.statusPop = uicontrol('Style','popupmenu','Parent',S.fig,'Units','normalized', ...
    'Position',[0.117 0.915 0.098 0.03],'String',statusOpts, ...
    'TooltipString','qc_status (critical/minor/clean)','Enable',tf2onoff(hasStatus), ...
    'Callback',@(h,~)onFilter());
S.list = uicontrol('Style','listbox','Parent',S.fig,'Units','normalized', ...
    'Position',[0.015 0.12 0.20 0.79],'FontName','Courier New','FontSize',9, ...
    'Callback',@(h,~)onListPick(h));

% banner (top)
S.banner = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.235 0.945 0.755 0.045],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',12,'FontWeight','bold');

% axes (centre)
S.ax = axes('Parent',S.fig,'Units','normalized','Position',[0.265 0.44 0.52 0.48]);
box(S.ax,'on'); grid(S.ax,'on'); hold(S.ax,'on');

% horizontal scroll bar under the trace: pans the (zoomed) view left/right across
% the whole plotted span, so a P-onset / T-offset that sits outside the current
% window can be scrolled into view. Only active when zoomed in (view < full span).
S.xslider = uicontrol('Style','slider','Parent',S.fig,'Units','normalized', ...
    'Position',[0.265 0.352 0.52 0.024],'Min',0,'Max',1,'Value',0, ...
    'Callback',@(~,~)onXScroll());
try
    S.xsliderL = addlistener(S.xslider,'ContinuousValueChange',@(~,~)onXScroll());
catch
end

% class/flag hint under the trace
S.hint = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.265 0.245 0.52 0.07],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',9,'ForegroundColor',[0.55 0.30 0.0]);

% fiducial list (right)
uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.80 0.90 0.19 0.025],'String','Fiducials (click to select)', ...
    'HorizontalAlignment','left','BackgroundColor','w','FontWeight','bold');
S.fidList = uicontrol('Style','listbox','Parent',S.fig,'Units','normalized', ...
    'Position',[0.80 0.55 0.19 0.35],'FontName','Courier New','FontSize',10, ...
    'Callback',@(h,~)onFidPick(h));
S.bm = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.80 0.32 0.19 0.21],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontName','Courier New','FontSize',10);

% buttons
mkbtn(0.265,'Prev',      @(~,~)go(-1));
mkbtn(0.345,'Next',      @(~,~)go(+1));
mkbtn(0.425,'Reset',     @(~,~)resetUnit());
mkbtn(0.505,'P absent',  @(~,~)toggleWave(S.PGRP,'p'));
mkbtn(0.585,'T absent',  @(~,~)toggleWave(S.TGRP,'t'));
% Q, R and S are single landmarks, so each gets its own absent toggle rather
% than a group toggle. Pressing one clears that landmark, pressing it again
% restores the worklist value.
mkbtnN(0.665,'Q absent', @(~,~)toggleLandmark(S.QPK));
mkbtnN(0.718,'R absent', @(~,~)toggleLandmark(S.RPK));
mkbtnN(0.771,'S absent', @(~,~)toggleLandmark(S.SPK));
S.delBtn = uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
    'Position',[0.825 0.175 0.075 0.05],'String','Delete rec','FontSize',9, ...
    'FontWeight','bold','ForegroundColor',[0.75 0 0],'Callback',@(~,~)deleteRecord());

S.status = uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.265 0.135 0.72 0.03],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',10,'FontWeight','bold', ...
    'ForegroundColor',[0.10 0.50 0.10],'String','All changes saved');

uicontrol('Style','text','Parent',S.fig,'Units','normalized', ...
    'Position',[0.265 0.005 0.72 0.085],'HorizontalAlignment','left', ...
    'BackgroundColor','w','FontSize',8,'String', ...
    ['Drag or click a fiducial to move it, and the arrow keys nudge it (Shift takes bigger steps).  ' ...
     'Every change is saved automatically, and edited units are marked with * in the list.  ' ...
     'Press n or p to change unit, and Delete to clear the selected landmark.  ' ...
     'Zoom: scroll wheel over the trace (or + / -). Pan with the scroll bar under the trace (or , / .), and press f to reset the view.  ' ...
     'The grey traces are the other 11 leads, scaled to this one over the visible window. Press o to hide or show them.  ' ...
     'The orange curve in the strip below is the spatial magnitude over the 8 independent leads, and the dotted line is its noise floor. ' ...
     'Read an onset as the first sustained lift off that floor and an offset as the last return to it. Press m to hide or show the strip.  ' ...
     'Press v for the teal velocity strip below it, which is sharper at the QRS boundaries and weaker at the T offset.  ' ...
     'Q, R and S have their own absent buttons. Use them when the complex has no such wave, for example no R in a QS complex or no Q in an rS complex.  ' ...
     'Press r on EVERY record when you are done with it, including ones you leave unchanged.']);

guidata(S.fig, S);
renderList();
openUnit();

% ===================================================================
%                          nested functions
% ===================================================================
    function mkbtn(x,label,cb)
        uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
            'Position',[x 0.175 0.075 0.05],'String',label,'FontSize',9,'Callback',cb);
    end

    % narrow variant, so the three QRS peak toggles fit in the space left
    % between the T absent button and the delete button
    function mkbtnN(x,label,cb)
        uicontrol('Style','pushbutton','Parent',S.fig,'Units','normalized', ...
            'Position',[x 0.175 0.050 0.05],'String',label,'FontSize',7.5, ...
            'Callback',cb);
    end

    function k = unitKey(ri)
        k = char(S.T.record_id(ri) + "|" + S.T.lead(ri) + "|" + string(S.T.beat_id(ri)));
    end

    % ---- left list ---------------------------------------------------
    function renderList()
        S = guidata(S.fig);
        nv = numel(S.view);
        cap = min(nv, S.LISTCAP);            % never build 200k strings at once
        strs = cell(cap + (nv>cap),1);
        for i = 1:cap
            ri = S.view(i);
            done = '  ';
            if isKey(S.corr, unitKey(ri))
                done = ' *';
            end
            if S.hasStatus; st = cstr(S.T.qc_status(ri)); else; st = ''; end
            strs{i} = sprintf('%s %-4s %-7s %-8s %s', done, cstr(S.T.lead(ri)), ...
                cstr(S.T.disease_class(ri)), st, cstr(S.T.flags(ri)));
        end
        if nv > cap
            strs{end} = sprintf('... (+%d more — narrow the filter)', nv-cap);
        end
        if isempty(strs); strs = {'(no units match filter)'}; end
        set(S.list,'String',strs,'Value',min(max(1,S.pos),max(1,numel(strs))));
        guidata(S.fig,S);
    end

    function onListPick(h)
        S = guidata(S.fig);
        if S.dirty; saveCorr(true); S = guidata(S.fig); end
        v = get(h,'Value');
        if v <= min(numel(S.view), S.LISTCAP)   % ignore the "+N more" sentinel row
            S.pos = v; guidata(S.fig,S); openUnit();
        else
            guidata(S.fig,S);
        end
    end

    % ---- combined class + qc_status filter ---------------------------
    function onFilter()
        S = guidata(S.fig);
        if S.dirty; saveCorr(true); S = guidata(S.fig); end
        co = get(S.classPop,'String');  S.classFilter  = co{get(S.classPop,'Value')};
        so = get(S.statusPop,'String'); S.statusFilter = so{get(S.statusPop,'Value')};
        mask = true(height(S.T),1);
        if ~strcmp(S.classFilter,'all')
            mask = mask & (S.T.disease_class == string(S.classFilter));
        end
        if S.hasStatus && ~strcmp(S.statusFilter,'all')
            mask = mask & (S.T.qc_status == string(S.statusFilter));
        end
        if ~isempty(keys(S.deleted))
            mask = mask & ~ismember(S.T.record_id, string(keys(S.deleted)));
        end
        S.view = find(mask).';
        S.pos = 1;
        guidata(S.fig,S);
        renderList();
        if ~isempty(S.view); openUnit(); end
    end

    function go(delta)
        S = guidata(S.fig);
        if S.dirty; saveCorr(true); S = guidata(S.fig); end
        S.pos = min(max(1, S.pos + delta), numel(S.view));
        guidata(S.fig,S);
        openUnit();
    end

    % ---- open a unit -------------------------------------------------
    function openUnit()
        S = guidata(S.fig);
        if isempty(S.view); return; end
      try
        ri = S.view(S.pos);
        key = unitKey(ri);

        % working fiducials from saved correction, else from ECGdeli row
        if isKey(S.corr, key)
            S.fv = S.corr(key).fv;
        else
            S.fv = arrayfun(@(k) getNum(S.T.(S.FKEYS{k})(ri)), 1:numel(S.FKEYS));
        end
        S.sel = 0;

        % load / cache the signal
        rec = cstr(S.T.record_id(ri));
        if ~isKey(S.sig, rec)
            M = loadSignal(cstr(S.T.path_raw(ri)));
            S.sig(rec) = M;
        end
        guidata(S.fig,S);

        % Window on THIS beat only, bounded three ways so it can never show a
        % second QRS and never clips the beat: (1) this beat's own P-onset and
        % T-offset drive the edges, (2) the mid-RR beat_start/beat_end clip toward
        % the detected neighbours (handles fast rhythms), and (3) physiological
        % caps around R (about R-0.5 s .. R+0.6 s) reject a landmark mislabelled
        % onto a neighbour and bound the view when ECGdeli missed a beat. A stray
        % landmark still appears in the right panel and can be clicked back in.
        n   = getNum(S.T.n_samples(ri));
        fsv = getNum(S.T.fs_hz(ri)); if isempty(fsv)||fsv==0||isnan(fsv); fsv = 500; end
        Rv  = getNum(S.T.r_peak_sample(ri));
        if isnan(Rv)
            fpv = S.fv(~isnan(S.fv));
            if ~isempty(fpv); Rv = median(fpv); else; Rv = 0; end
        end
        loCap = Rv - round(0.50*fsv);  hiCap = Rv + round(0.70*fsv);
        bs = getNum(S.T.beat_start_sample(ri)); if isnan(bs); bs = loCap; end
        be = getNum(S.T.beat_end_sample(ri));   if isnan(be); be = hiCap; end
        ctxL = round(0.10*fsv);         % baseline shown BEFORE P-onset (~100 ms)
        ctxR = round(0.18*fsv);         % baseline shown AFTER  T-offset (~180 ms) so a
                                        % T-offset ECGdeli placed too EARLY still shows the
                                        % real T tail to its right without needing to scroll
        % Auto-fit view. Priority is the CURRENT beat's own P-onset and T-offset:
        % these belong to this beat, so they must never be clipped, even when the
        % mid-RR guard (bs/be) or a long PR/QT would otherwise crop them. min()/max()
        % lets bs/be push the edges OUT for context but never IN past the fiducials.
        Pon = S.fv(1);   if isnan(Pon)  || Pon  < loCap || Pon  > Rv; Pon  = max(loCap, bs); end
        Toff = S.fv(11); if isnan(Toff) || Toff > hiCap || Toff < Rv; Toff = min(hiCap, be); end
        lo = min(max(bs, loCap), Pon  - ctxL);
        hi = max(min(be, hiCap), Toff + ctxR);
        if hi <= lo; lo = Rv - round(0.30*fsv); hi = Rv + round(0.40*fsv); end
        lo = max(0, lo);  hi = min(n-1, hi);
        % Plot span (S.win): the auto-fit view plus ~0.30 s of margin on each side. That
        % is enough to scroll a clipped or mis-placed P-onset / T-offset into view, but
        % NOT so wide that whole neighbouring beats appear (the earlier full-beat margin
        % showed ~3 beats). The default view still opens tight on THIS beat; scroll/zoom
        % reveals the margin. Bounded to the signal so it can't run past the record.
        mrg     = round(0.30*fsv);
        S.win   = [max(0, lo - mrg), min(n-1, hi + mrg)];
        S.xview = [lo, hi] * (1000/fsv);   % current visible x-limits, in ms
        S.xfit  = S.xview;                 % remembered auto-fit (the 'f' key restores it)
        guidata(S.fig,S);

        % banner + hint
        cls = cstr(S.T.disease_class(ri));
        set(S.banner,'String',sprintf('%s   |   lead %s   |   %s   |   beat %d   |   unit %d/%d', ...
            cstr(S.T.record_id(ri)), cstr(S.T.lead(ri)), upper(cls), ...
            getNum(S.T.beat_id(ri)), S.pos, numel(S.view)));
        flg = cstr(S.T.flags(ri));
        hstr = ['Flag: ' flg];
        fix = flagFix(flg);
        if ~isempty(fix); hstr = [hstr '   =>  ' fix]; end
        if isKey(S.HINT, cls); hstr = [hstr '    -    ' S.HINT(cls)]; end
        if ismember('wide_window', S.T.Properties.VariableNames) && getNum(S.T.wide_window(ri)) == 1
            hstr = [hstr '    -    Note. This recording appears to have an undetected beat next to this one.'];
        end
        set(S.hint,'String',hstr);

        drawUnit();
        syncListSel();          % just move the selection, the list itself is only
                                % rebuilt on filter/review change (keeps Next/Prev fast)
      catch mErr
        try; set(S.banner,'String',['Render error on this unit (skipped): ' mErr.message]); catch; end
        try; renderList(); catch; end
      end
    end

    % move the left-list highlight to the current unit without rebuilding it
    function syncListSel()
        S = guidata(S.fig);
        if isempty(S.view) || ~isgraphics(S.list); return; end
        v = min(max(1, S.pos), min(numel(S.view), S.LISTCAP));
        set(S.list, 'Value', v);
    end

    % ---- draw --------------------------------------------------------
    function drawUnit()
        S = guidata(S.fig);
        ri = S.view(S.pos);
        rec = cstr(S.T.record_id(ri));
        M = S.sig(rec);
        li = find(strcmp(S.LEADS, cstr(S.T.lead(ri))), 1);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs = 500; end
        msps = 1000/fs;

        cla(S.ax); hold(S.ax,'on'); grid(S.ax,'on');
        s = round(S.win(1)); e = round(S.win(2));
        if isempty(M) || isnan(li)
            text(0.5,0.5,'Signal not loaded — check the dataset root / path\_raw', ...
                'Parent',S.ax,'Units','normalized','HorizontalAlignment','center', ...
                'Color',[0.6 0 0]);
            S.hLine = gobjects(1,numel(S.FKEYS)); guidata(S.fig,S); updatePanel(); return
        end
        idxRange = (s:e) + 1;                     % 1-based MATLAB indices
        idxRange = idxRange(idxRange>=1 & idxRange<=size(M,2));
        tms = (idxRange-1) * msps;                % sample(0-based)*ms
        V   = M(li, idxRange);

        % resolve the visible x-window now, so the y-scale can auto-fit whatever
        % the current zoom shows (zoom into the P and its small wave fills the axis)
        xv = S.xview;
        if isempty(xv) || any(~isfinite(xv)) || xv(2) <= xv(1); xv = tms([1 end]); end
        xv(1) = max(xv(1), tms(1)); xv(2) = min(xv(2), tms(end));
        if xv(2) <= xv(1); xv = tms([1 end]); end
        visMask = tms >= xv(1) & tms <= xv(2);
        if ~any(visMask); visMask = true(size(tms)); end
        vis = V(visMask);
        if isempty(vis); vis = V; end

        % The y-limits are resolved here rather than after plotting, so the magnitude
        % strip can be given room below the trace before anything is drawn. ytr keeps
        % the band the trace itself occupies, which is what the fiducial labels use.
        yl = [min(vis) max(vis)];
        if ~all(isfinite(yl)) || diff(yl)==0; yl = [-1 1]; end
        pad = 0.10*diff(yl); yl = yl + [-pad pad];
        ytr = yl;

        % ---- spatial magnitude across the independent leads (GOLD) -------
        % Root sum of squares over I, II and V1 to V6 at every sample, after each lead
        % has had its own baseline removed. This answers the question the twelve-trace
        % overlay can only hint at, which is where the heart actually starts and stops.
        %
        % A lead records the projection of the cardiac dipole onto its own axis, so a
        % lead lying near perpendicular to the initial depolarisation vector registers
        % nothing for the first several milliseconds and reports an onset that is late.
        % The error only ever runs one way. A lead can make an onset look later and an
        % offset look earlier, never the reverse, so every one of the twelve is biased
        % inward and a p25 or p75 reconciliation across them inherits that bias rather
        % than cancelling it.
        %
        % Squaring removes the polarity problem outright, an inverted lead contributes
        % exactly as much as an upright one. Summing over eight axes removes the
        % perpendicularity problem, since a vector perpendicular to one lead is not
        % perpendicular to all eight. What is left is a single curve that rests on its
        % noise floor and lifts the moment any part of the heart depolarises.
        magOn = isfield(S,'magnitude') && S.magnitude;
        velOn = isfield(S,'velocity')  && S.velocity;
        mag = []; magFloor = NaN; vel = []; velFloor = NaN;
        if (magOn || velOn) && isfield(S,'MAGLEADS')
            gi = [];
            for j = 1:numel(S.MAGLEADS)
                k2 = find(strcmp(S.LEADS, S.MAGLEADS{j}), 1);
                if ~isempty(k2) && k2 <= size(M,1); gi(end+1) = k2; end %#ok<AGROW>
            end
            if numel(gi) >= 4
                D = M(gi, idxRange);
                % Baseline per lead over the WHOLE beat window and not over the visible
                % one, so the curve keeps its shape as the reviewer zooms. A baseline
                % that moved with the zoom would make the lift-off point move with it.
                D = D - median(D, 2, 'omitnan');
                D(~isfinite(D)) = 0;
                if magOn
                    mag = sqrt(sum(D.^2, 1));
                    magFloor = qtile(mag, 0.10);
                end
                if velOn && size(D,2) >= 3
                    % Central first difference, one sample either side. The derivative
                    % of a constant is zero, so a lead that is merely offset from its
                    % baseline contributes nothing here, and the curve reports motion
                    % alone. That is what makes its foot at the QRS so much shorter
                    % than the amplitude foot, and also what makes it fade at the T
                    % offset, where the wave ends by flattening rather than by moving.
                    G = zeros(size(D));
                    G(:,2:end-1) = 0.5*(D(:,3:end) - D(:,1:end-2));
                    G(:,1)   = D(:,2)   - D(:,1);
                    G(:,end) = D(:,end) - D(:,end-1);
                    vel = sqrt(sum(G.^2, 1));
                    velFloor = qtile(vel, 0.10);
                end
            end
        end
        % Each active curve gets its own band under the trace, stacked downward in the
        % order magnitude then velocity, so the two feet can be read against the same
        % fiducial line without either curve being drawn over the other. With both on
        % the bands are made shorter so the trace itself keeps most of the height.
        strips = {};
        if ~isempty(mag); strips{end+1} = {mag, magFloor, [0.85 0.42 0.10], ' |v| '}; end %#ok<AGROW>
        if ~isempty(vel); strips{end+1} = {vel, velFloor, [0.00 0.48 0.50], ' |dv| '}; end %#ok<AGROW>
        for si = 1:numel(strips)
            cur = strips{si};
            cvec = cur{1}; cfl = cur{2}; ccol = cur{3}; clab = cur{4};
            hband = 0.34 * diff(ytr);
            if numel(strips) > 1; hband = 0.26 * diff(ytr); end
            yb = yl(1) - hband;
            yl(1) = yb;
            % Normalised over the visible window, on the same principle as the overlay,
            % so zooming into the P wave rescales the curve and makes a small P onset
            % readable next to a QRS that is twenty times its size.
            cvis = cvec(visMask); if isempty(cvis); cvis = cvec; end
            ctop = max(cvis); if ~isfinite(ctop) || ctop <= 0; ctop = 1; end
            cY = yb + 0.06*hband + (cvec/ctop) * (0.86*hband);
            plot(S.ax, tms([1 end]), (yb+hband)*[1 1], '-','Color',[0.88 0.88 0.90], 'LineWidth',0.5);
            if isfinite(cfl)
                fy = yb + 0.06*hband + (cfl/ctop) * (0.86*hband);
                plot(S.ax, tms([1 end]), [fy fy], ':', 'Color',ccol, 'LineWidth',0.8);
            end
            plot(S.ax, tms, min(cY, yb+hband), '-', 'Color',ccol, 'LineWidth',1.3);
            text(tms(1), yb+0.90*hband, clab, 'Parent',S.ax, 'Color',ccol, ...
                'FontSize',8, 'FontWeight','bold', 'HorizontalAlignment','left', ...
                'VerticalAlignment','middle', 'Clipping','on');
        end

        % ---- 12-lead overlay (GOLD) --------------------------------------
        % Drawn BEFORE the reference lead so the reference stays on top, and
        % scaled per lead so a small-amplitude lead is still legible. The scale
        % is taken over the VISIBLE window only, so zooming into the P wave
        % rescales every overlay lead to that window and the P onset of the
        % earliest lead becomes readable. Centring uses the median of the
        % visible window rather than the mean, which keeps a large QRS from
        % dragging the baseline of the trace off-centre. M is already the full
        % 12 x N matrix held in S.sig, so this costs no extra file reads.
        if isfield(S,'overlay') && S.overlay
            pC = median(vis,'omitnan'); pA = max(abs(vis - pC));
            if ~isfinite(pA) || pA == 0; pA = 1; end
            for j = 1:size(M,1)
                if j == li; continue; end
                W  = M(j, idxRange);
                Wv = W(visMask);
                wC = median(Wv,'omitnan'); wA = max(abs(Wv - wC));
                if ~isfinite(wA) || wA == 0; continue; end
                Wn = (W - wC) * (pA/wA) + pC;
                plot(S.ax, tms, Wn, '-', 'Color',[0.62 0.66 0.72], 'LineWidth',0.6);
            end
        end
        plot(S.ax, tms, V, '-', 'Color',[0.10 0.20 0.45], 'LineWidth',1.6);
        if ytr(1)<0 && ytr(2)>0; plot(S.ax, tms([1 end]), [0 0], '-','Color',[0.8 0.8 0.8]); end

        S.hLine = gobjects(1,numel(S.FKEYS));
        S.hMark = gobjects(1,numel(S.FKEYS));
        % Labels are staggered over the TRACE band only, so they never drift down into
        % the magnitude strip. The fiducial lines themselves still span the full axis,
        % which is the point of the strip, since the line crosses the magnitude curve
        % and shows at a glance whether the landmark sits at lift-off or after it.
        yr = diff(ytr); nT = 6; shown = 0;        % stagger labels over nT height tiers
        for k = 1:numel(S.FKEYS)
            fv = S.fv(k);
            if isnan(fv) || fv < s || fv > e; continue; end
            xk = fv * msps;
            sel = (k==S.sel);
            lwk = 1.3; if sel; lwk = 3.2; end
            S.hLine(k) = plot(S.ax, [xk xk], yl, '-', 'Color',S.FCOL(k,:), 'LineWidth',lwk);
            vy = M(li, min(max(round(fv)+1,1),size(M,2)));
            S.hMark(k) = plot(S.ax, xk, vy, 'o', 'MarkerSize',7, 'LineWidth',1.5, ...
                'MarkerEdgeColor',S.FCOL(k,:), 'MarkerFaceColor','w');
            % staggered, horizontal, boxed label (readable even when landmarks cluster)
            tier = mod(shown, nT); shown = shown + 1;
            ylab = ytr(2) - 0.04*yr - tier*(0.92*yr/nT);
            fw = 'bold'; if sel; fw = 'bold'; end
            text(xk, ylab, [' ' S.FLABEL{k} ' '], 'Parent',S.ax, 'Color',S.FCOL(k,:), ...
                'FontSize',8.5, 'FontWeight',fw, 'BackgroundColor',[1 1 1], ...
                'EdgeColor',S.FCOL(k,:), 'Margin',1, 'LineWidth',0.5, ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', 'Clipping','on');
        end
        % x-limits follow the (zoom/pan-adjustable) view window resolved above, not
        % the plot span, so a zoom the user set survives a fiducial edit / redraw.
        ylim(S.ax, yl); xlim(S.ax, xv);
        xlabel(S.ax,'time (ms)'); ylabel(S.ax,'voltage');
        guidata(S.fig,S);
        updatePanel(); refreshStatus(); syncXSlider();
    end

    % ---- right panel fiducial list + biomarkers ---------------------
    function updatePanel()
        S = guidata(S.fig);
        ri = S.view(S.pos);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs=500; end
        msps = 1000/fs;
        items = cell(numel(S.FKEYS),1);
        for k = 1:numel(S.FKEYS)
            if isnan(S.fv(k))
                items{k} = sprintf('%-7s   absent', S.FLABEL{k});
            else
                items{k} = sprintf('%-7s %5d  %5.0f ms', S.FLABEL{k}, ...
                    round(S.fv(k)), S.fv(k)*msps);
            end
        end
        set(S.fidList,'String',items,'Value',max(1,S.sel));

        pr  = ivInterval(S.fv(4),  S.fv(1), msps);   % QRS_on - P_on
        qrs = ivInterval(S.fv(8),  S.fv(4), msps);   % QRS_off - QRS_on
        qt  = ivInterval(S.fv(11), S.fv(4), msps);   % T_off - QRS_on
        td  = ivInterval(S.fv(11), S.fv(9), msps);   % T_off - T_on
        lines = {
            bmLine('PR', pr, 80, 400), ...
            bmLine('QRS', qrs, 40, 200), ...
            bmLine('QT', qt, 250, 700), ...
            bmLine('T dur', td, 60, 320), '', ...
            sprintf('P present : %d', any(~isnan(S.fv(S.PGRP)))), ...
            sprintf('T present : %d', any(~isnan(S.fv(S.TGRP)))), ...
            sprintf('Q R S     : %d %d %d', ~isnan(S.fv(S.QPK)), ...
                    ~isnan(S.fv(S.RPK)), ~isnan(S.fv(S.SPK)))};
        % ordering check
        seq = S.fv(~isnan(S.fv));
        if any(diff(seq) < 0); lines{end+1} = 'WARNING: out of order'; end
        set(S.bm,'String',lines);
        guidata(S.fig,S);
    end

    % ---- mouse -------------------------------------------------------
    function onDown(~,~)
        S = guidata(S.fig);
        if isempty(S.view); return; end
        cp = get(S.ax,'CurrentPoint'); x = cp(1,1);
        xl = xlim(S.ax);
        if x < xl(1) || x > xl(2); return; end
        ri = S.view(S.pos);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs=500; end
        msps = 1000/fs;
        xs = S.fv * msps;                          % present fiducial x (ms), NaN for absent
        d = abs(xs - x);
        [dmin, k] = min(d);
        tol = 0.03 * (xl(2)-xl(1));
        if ~isnan(dmin) && dmin <= tol
            S.dragF = k; S.sel = k; guidata(S.fig,S); drawUnit();
        elseif S.sel > 0
            guidata(S.fig,S);
            setFid(S.sel, x/msps);                 % move/restore selected
            saveCorr(true);
        end
    end

    function onMotion(~,~)
        S = guidata(S.fig);
        if S.dragF == 0; return; end
        cp = get(S.ax,'CurrentPoint'); x = cp(1,1);
        ri = S.view(S.pos);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs=500; end
        setFid(S.dragF, x*fs/1000);
    end

    function onUp(~,~)
        S = guidata(S.fig);
        if S.dragF ~= 0; S.dragF = 0; guidata(S.fig,S); saveCorr(true); end  % persist drag
    end

    % ---- keyboard ----------------------------------------------------
    function onKey(~,ev)
        S = guidata(S.fig);
        if isempty(S.view); return; end
        shift = any(strcmp(ev.Modifier,'shift'));
        step = 1; if shift; step = 5; end
        switch ev.Key
            case 'leftarrow'
                if S.sel>0 && ~isnan(S.fv(S.sel)); setFid(S.sel, S.fv(S.sel)-step); saveCorr(true); end
            case 'rightarrow'
                if S.sel>0 && ~isnan(S.fv(S.sel)); setFid(S.sel, S.fv(S.sel)+step); saveCorr(true); end
            case 'o'
                S.overlay = ~S.overlay; guidata(S.fig,S); drawUnit();
            case 'm'
                S.magnitude = ~S.magnitude; guidata(S.fig,S); drawUnit();
            case 'v'
                S.velocity = ~S.velocity; guidata(S.fig,S); drawUnit();
            case 'r'; markReviewed();
            case 'n'; go(+1);
            case 'p'; go(-1);
            case 's'; saveCorr(false);
            case 'delete';    clearSel();
            case 'backspace'; clearSel();
            % --- zoom / pan the x-axis (view only, never touches the fiducials) ---
            case {'equal','add'};      zoomView(1/1.4);   % + : zoom in
            case {'hyphen','subtract'};zoomView(1.4);     % - : zoom out
            case {'comma'};            panView(-0.25);    % , : pan left
            case {'period'};           panView(+0.25);    % . : pan right
            case {'f','0'};            fitView();         % f / 0 : reset to auto-fit
        end
    end

    % ---- view zoom / pan (x-axis only) -------------------------------
    % These change S.xview (the visible window) and redraw. They never change
    % S.fv, so labels stay put. the y-scale auto-fits whatever is now visible.
    function setView(xv)
        S = guidata(S.fig);
        ri = S.view(S.pos);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs=500; end
        lo = (S.win(1)) * (1000/fs);  hi = (S.win(2)) * (1000/fs);   % plot span in ms
        w  = xv(2) - xv(1);
        w  = max(w, 30);                          % never zoom tighter than ~30 ms
        if w > (hi-lo); w = hi-lo; end
        c  = mean(xv);
        a  = c - w/2;  b = c + w/2;
        if a < lo; a = lo; b = lo + w; end
        if b > hi; b = hi; a = hi - w; end
        S.xview = [a, b];
        guidata(S.fig,S); drawUnit();
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
        guidata(S.fig,S); drawUnit();
    end

    % ---- scroll-wheel zoom, centred on the cursor --------------------
    function onScroll(~,ev)
        S = guidata(S.fig);
        if isempty(S.view); return; end
        cp = get(S.ax,'CurrentPoint'); xc = cp(1,1);
        xv = S.xview;
        if xc < xv(1) || xc > xv(2); xc = mean(xv); end   % cursor off-axis -> centre
        if ev.VerticalScrollCount > 0; f = 1.4; else; f = 1/1.4; end   % down=out, up=in
        % zoom about the cursor: keep xc at the same fractional position
        r  = (xc - xv(1)) / max(eps, (xv(2)-xv(1)));
        w  = (xv(2)-xv(1)) * f;
        setView([xc - r*w, xc + (1-r)*w]);
    end

    % ---- horizontal scroll bar --------------------------------------
    % Dragging the bar moves the view's LEFT edge across the plot span, keeping
    % the current zoom width. onXScroll reacts to the user. syncXSlider pushes the
    % view state back onto the bar (guarded so the two don't loop).
    function onXScroll()
        S = guidata(S.fig);
        if S.xslBusy || isempty(S.view) || ~isgraphics(S.xslider); return; end
        if ~strcmp(get(S.xslider,'Enable'),'on'); return; end
        a = get(S.xslider,'Value');                 % new left edge (ms)
        w = S.xview(2) - S.xview(1);
        S.xview = [a, a + w];
        guidata(S.fig,S); drawUnit();
    end
    function syncXSlider()
        S = guidata(S.fig);
        if ~isgraphics(S.xslider); return; end
        ri = S.view(S.pos);
        fs = getNum(S.T.fs_hz(ri)); if isempty(fs)||fs==0||isnan(fs); fs=500; end
        pLo = S.win(1)*(1000/fs);  pHi = S.win(2)*(1000/fs);   % plot span (ms)
        w   = S.xview(2) - S.xview(1);
        S.xslBusy = true; guidata(S.fig,S);
        if w >= (pHi - pLo) - 1                     % fully zoomed out -> nothing to scroll
            set(S.xslider,'Min',-1e9); set(S.xslider,'Max',1);
            set(S.xslider,'Min',0); set(S.xslider,'Value',0,'Enable','off');
        else
            minL = pLo;  maxL = pHi - w;
            val  = min(max(S.xview(1), minL), maxL);
            span = max(maxL - minL, eps);
            smallStep = min(max((0.15*w)/span, 0.01), 0.999);
            bigStep   = min(max(w/span,        0.05), 0.999);
            % drop Min far below any prior Max first, so no set ever hits Min>=Max
            set(S.xslider,'Min',-1e9);
            set(S.xslider,'Max',maxL);
            set(S.xslider,'Min',minL);
            set(S.xslider,'Value',val,'SliderStep',[smallStep bigStep],'Enable','on');
        end
        S.xslBusy = false; guidata(S.fig,S);
    end

    % ---- edits -------------------------------------------------------
    function setFid(k, newFvSamples)
        S = guidata(S.fig);
        ri = S.view(S.pos);
        n  = getNum(S.T.n_samples(ri));
        v  = min(max(0, round(newFvSamples)), n-1);
        S.fv(k) = v;
        S.dirty = true;
        guidata(S.fig,S);
        stashWorking();
        drawUnit();
    end

    function clearSel()
        S = guidata(S.fig);
        if S.sel <= 0; return; end
        S.fv(S.sel) = NaN; S.dirty = true;
        guidata(S.fig,S); stashWorking(); drawUnit(); saveCorr(true);
    end

    function toggleWave(grp, which) %#ok<INUSD>
        S = guidata(S.fig);
        if all(isnan(S.fv(grp)))
            % restore from ECGdeli row
            ri = S.view(S.pos);
            for k = grp; S.fv(k) = getNum(S.T.(S.FKEYS{k})(ri)); end
        else
            S.fv(grp) = NaN;                       % mark whole wave absent
        end
        S.dirty = true;
        guidata(S.fig,S); stashWorking(); drawUnit(); saveCorr(true);  % toggle -> CSV
    end

    % Toggle one landmark between absent and its worklist sample. Used for the
    % Q, R and S buttons, where the wave either exists or it does not and there
    % is no onset or offset to clear alongside it.
    function toggleLandmark(k)
        S = guidata(S.fig);
        if isnan(S.fv(k))
            ri = S.view(S.pos);
            S.fv(k) = getNum(S.T.(S.FKEYS{k})(ri));   % restore from the worklist
        else
            S.fv(k) = NaN;                            % this wave does not exist
        end
        S.sel = k;
        S.dirty = true;
        guidata(S.fig,S); stashWorking(); drawUnit(); saveCorr(true);
    end

    function resetUnit()
        S = guidata(S.fig);
        ri = S.view(S.pos);
        S.fv = arrayfun(@(k) getNum(S.T.(S.FKEYS{k})(ri)), 1:numel(S.FKEYS));
        S.dirty = true;
        guidata(S.fig,S); stashWorking(); drawUnit(); saveCorr(true);
    end

    % stash working values into corrections map (touched, not reviewed)
    function stashWorking()
        S = guidata(S.fig);
        key = unitKey(S.view(S.pos));
        rec = struct();
        if isKey(S.corr,key); rec = S.corr(key); end
        if ~isfield(rec,'reviewed'); rec.reviewed = false; end
        rec.fv = S.fv;
        rec.edited_at = nowISO();
        S.corr(key) = rec;
        guidata(S.fig,S);
        markEditedInList();
    end

    % mark the current unit's row in the left list as edited, without a full rebuild
    function markEditedInList()
        S = guidata(S.fig);
        if ~isgraphics(S.list) || isempty(S.view); return; end
        v = S.pos;
        if v >= 1 && v <= min(numel(S.view), S.LISTCAP)
            strs = get(S.list,'String');
            if iscell(strs) && v <= numel(strs) && numel(strs{v}) >= 2
                ln = strs{v}; ln(1:2) = ' *'; strs{v} = ln;
                set(S.list,'String',strs);
            end
        end
    end

    % ---- panel selection --------------------------------------------
    function onFidPick(h)
        S = guidata(S.fig);
        S.sel = get(h,'Value');
        guidata(S.fig,S); drawUnit();
    end

    % ---- review + save ----------------------------------------------
    function markReviewed()
        S = guidata(S.fig);
        stashWorking(); S = guidata(S.fig);
        key = unitKey(S.view(S.pos));
        rec = S.corr(key); rec.reviewed = true; rec.edited_at = nowISO();
        S.corr(key) = rec; S.dirty = true;
        guidata(S.fig,S);
        saveCorr(true);
        renderList();      % refresh so this unit's 'OK' marker appears in the list
        go(+1);
    end

    function saveCorr(quiet)
        S = guidata(S.fig);
        writeCorrections(S.outCsv, S.corr, S.T, S.FKEYS, S.PGRP, S.QGRP, S.TGRP);
        S.dirty = false;
        guidata(S.fig,S);
        refreshStatus();
        if ~quiet
            fprintf('Saved corrections -> %s\n', S.outCsv);
        end
    end

    function refreshStatus()
        S = guidata(S.fig);
        if ~isfield(S,'status') || ~isgraphics(S.status); return; end
        nrev = numel(keys(S.corr));
        if S.dirty
            set(S.status,'ForegroundColor',[0.75 0.10 0.10], 'String', ...
                sprintf('Saving...  -  %d edited unit(s)', nrev));
        else
            set(S.status,'ForegroundColor',[0.10 0.50 0.10], 'String', ...
                sprintf('Changes saved  -  %d edited unit(s)', nrev));
        end
    end

    function onClose(~,~)
        try; S = guidata(S.fig); if S.dirty; saveCorr(true); end; catch; end
        delete(S.fig);
    end

    % ---- delete (exclude) the whole record --------------------------
    function deleteRecord()
        S = guidata(S.fig);
        if isempty(S.view); return; end
        ri  = S.view(S.pos);
        rec = cstr(S.T.record_id(ri));
        cls = cstr(S.T.disease_class(ri));
        ld  = cstr(S.T.lead(ri));
        q = questdlg(sprintf(['Exclude the WHOLE record %s (all 12 leads)?' char(10) ...
            'Recorded in deleted_records.csv and applied by apply_deletions.py. ' ...
            'Raw files are kept.'], rec), 'Delete record', 'Delete', 'Cancel', 'Cancel');
        if ~strcmp(q,'Delete'); return; end
        S.deleted(rec) = struct('disease_class',cls,'seen_on_lead',ld,'deleted_at',nowISO());
        guidata(S.fig,S);
        writeDeletions();
        S = guidata(S.fig);
        keep = S.T.record_id(S.view) ~= string(rec);
        S.view = S.view(keep);
        if S.pos > numel(S.view); S.pos = max(1, numel(S.view)); end
        guidata(S.fig,S);
        renderList();
        if ~isempty(S.view)
            openUnit();
        else
            set(S.banner,'String','(no units left in this filter)'); cla(S.ax);
        end
    end

    function writeDeletions()
        S = guidata(S.fig);
        fid = fopen(S.delCsv,'w');
        if fid < 0; warning('Cannot open %s for writing.', S.delCsv); return; end
        cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
        fprintf(fid,'record_id,disease_class,seen_on_lead,deleted_at\n');
        ks = keys(S.deleted);
        for a = 1:numel(ks)
            d = S.deleted(ks{a});
            fprintf(fid,'%s,%s,%s,%s\n', ks{a}, d.disease_class, d.seen_on_lead, d.deleted_at);
        end
    end

    % ---- signal loading ---------------------------------------------
    function M = loadSignal(pathRaw)
        % Do NOT re-read guidata here: S is a shared nested-function variable, so
        % reloading it would discard the S.fv that openUnit set just before this
        % call (openUnit saves S only after caching the signal). That clobber made
        % the first view of each record show its fiducials as absent until you
        % navigated away and back. S.root is already current in the shared S.
        full = resolveRaw(S.root, pathRaw);
        if isempty(full)
            warning('Cannot find raw signal: %s', pathRaw);
            M = []; return
        end
        M = readmatrix(full);
        if size(M,1) > size(M,2); M = M.'; end     % ensure leads x samples
    end

    function bl = bmLine(name, v, lo, hi)
        if isnan(v)
            bl = sprintf('%-6s : --', name);
        else
            flag = ''; if v<lo || v>hi; flag = ' !'; end
            bl = sprintf('%-6s : %4.0f ms%s', name, v, flag);
        end
    end
end

% ======================================================================
%                       plain helper functions
% ======================================================================
function v = ivInterval(fa, fb, m)
    if isnan(fa) || isnan(fb); v = NaN; else; v = (fa - fb) * m; end
end

function q = qtile(x, p)
%QTILE  Nearest-rank quantile of a vector, with no Statistics Toolbox dependency.
%   Used for the magnitude strip's noise floor. prctile lives in the Statistics
%   Toolbox and the tool must run without it, so the rank is taken directly.
    x = x(isfinite(x));
    if isempty(x); q = NaN; return; end
    x = sort(x(:));
    n = numel(x);
    q = x(min(max(round(p*(n-1))+1, 1), n));
end

function v = getNum(x)
%GETNUM  Scalar double from a table cell (string/char/numeric), NaN if blank.
    if isnumeric(x); v = double(x); if isempty(v); v = NaN; end; return; end
    s = strtrim(string(x));
    if s=="" || s=="None" || ismissing(s); v = NaN; else; v = str2double(s); end
end

function s = cstr(x)
%CSTR  Char row vector from a table value (string/char/categorical/numeric).
%   Returns '' for a <missing>/empty value so char() never errors on blanks
%   (readtable(...,'TextType','string') stores empty cells as <missing>).
    if isnumeric(x)
        if isempty(x) || all(isnan(x)); s = ''; else; s = num2str(x); end
        return
    end
    x = string(x);
    if isempty(x) || all(ismissing(x)); s = ''; else; s = char(x); end
end

function c = uniqueClasses(T)
    c = cellstr(unique(T.disease_class));
    c = c(:).';
end

function c = uniqueStatus(T)
%UNIQUESTATUS  qc_status values, ordered critical -> minor -> clean when present.
    u = cellstr(unique(T.qc_status));
    order = {'critical','minor','clean'};
    c = [order(ismember(order,u)), setdiff(u(:).', order)];
end

function s = tf2onoff(tf)
    if tf; s = 'on'; else; s = 'off'; end
end

function fix = flagFix(flagStr)
%FLAGFIX  One-line "what to check" suggestion parsed from a unit's qc_flags.
%   Mirrors MANUAL_LABELLING_PROTOCOL.md section 2. Empty if no known token.
    fix = '';
    f = lower(char(flagStr));
    if isempty(f); return; end
    parts = {};
    if contains(f,'pr=')
        parts{end+1} = 'PR flag. Is there really a P in this lead? If not, mark P absent, otherwise fix the P-onset (long PR is genuine in AV block)';
    end
    if contains(f,'qt=')
        parts{end+1} = 'QT flag. Re-place the T-offset by the tangent method (long QT is genuine in MI)';
    end
    if contains(f,'qrsdur=')
        parts{end+1} = 'QRS-dur flag. Move the QRS-onset or J-point to the true wide-complex bounds (wide QRS is genuine in LBBB/RBBB)';
    end
    if contains(f,'gross_boundary')
        parts{end+1} = 'Gross inversion. A landmark is out of order, so move it back into the P, QRS, T sequence';
    end
    fix = strjoin(parts, '   |   ');
end

function root = inferRepoRoot(batchCsv)
%INFERREPOROOT  Walk up from the batch file to a folder holding WP2_largeDataset_Noise.
    root = '';
    d = fileparts(batchCsv);
    for i = 1:8
        if exist(fullfile(d,'WP2_largeDataset_Noise'),'dir') == 7 || ...
           exist(fullfile(d,'config','paths.yaml'),'file') == 2
            root = d; return
        end
        nd = fileparts(d);
        if strcmp(nd,d); break; end
        d = nd;
    end
end

function full = resolveRaw(root, pathRaw)
%RESOLVERAW  Resolve a worklist path_raw against the dataset root.
    pathRaw = char(pathRaw);
    cand = { fullfile(root, pathRaw), ...
             fullfile(root, regexprep(pathRaw,'^[^/\\]+[/\\]','')), ...
             pathRaw };
    full = '';
    for i = 1:numel(cand)
        if exist(cand{i},'file') == 2; full = cand{i}; return; end
    end
end

function s = nowISO()
    s = datestr(now,'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>
end

function del = readDeletions(delCsv)
%READDELETIONS  Load deleted_records.csv into a record_id -> struct map (resume).
    del = containers.Map('KeyType','char','ValueType','any');
    if exist(delCsv,'file') ~= 2; return; end
    try
        D = readtable(delCsv,'TextType','string');
    catch
        return
    end
    if ~ismember('record_id', D.Properties.VariableNames); return; end
    for i = 1:height(D)
        r = char(D.record_id(i));
        dc = ''; sl = ''; da = '';
        if ismember('disease_class',D.Properties.VariableNames); dc = cstr(D.disease_class(i)); end
        if ismember('seen_on_lead', D.Properties.VariableNames); sl = cstr(D.seen_on_lead(i)); end
        if ismember('deleted_at',  D.Properties.VariableNames); da = cstr(D.deleted_at(i)); end
        del(r) = struct('disease_class',dc,'seen_on_lead',sl,'deleted_at',da);
    end
end

function corr = readCorrections(outCsv, FKEYS)
%READCORRECTIONS  Load an existing corrections CSV back into the map (resume).
    corr = containers.Map('KeyType','char','ValueType','any');
    try
        C = readtable(outCsv,'TextType','string');
    catch ME
        warning('medalcare:readcorr', ...
            ['Could not read existing corrections %s: %s. Starting with an ' ...
             'empty in-memory set; on-disk corrections are preserved because ' ...
             'saving now merges with the file instead of overwriting it.'], ...
            outCsv, ME.message);
        return
    end
    if ~all(ismember({'record_id','lead','beat_id'}, C.Properties.VariableNames)); return; end
    for i = 1:height(C)
        key = char(C.record_id(i) + "|" + C.lead(i) + "|" + string(C.beat_id(i)));
        fv = nan(1,numel(FKEYS));
        for k = 1:numel(FKEYS)
            if ismember(FKEYS{k}, C.Properties.VariableNames)
                fv(k) = getNum(C.(FKEYS{k})(i));
            end
        end
        rec = struct('fv',fv,'reviewed',false,'edited_at','');
        if ismember('reviewed',C.Properties.VariableNames); rec.reviewed = getNum(C.reviewed(i))==1; end
        if ismember('edited_at',C.Properties.VariableNames); rec.edited_at = cstr(C.edited_at(i)); end
        corr(key) = rec;
    end
end

function writeCorrections(outCsv, corr, T, FKEYS, PGRP, QGRP, TGRP)
%WRITECORRECTIONS  Merge the in-memory corrections with any rows already on
%   disk and write the union atomically. A save can only ADD or UPDATE a unit,
%   never drop one that was previously saved, so corrections cannot vanish if
%   the in-memory set is ever incomplete. Schema is the one
%   merge_manual_corrections.py consumes.
    cols = [{'record_id','disease_class','lead','beat_id','fs_hz','n_samples'}, ...
            FKEYS, {'p_present','qrs_present','q_present','r_present','s_present', ...
            't_present','flags', ...
            'also_delineator','priority','label_source','reviewed','edited_at'}];
    headerLine = strjoin(cols, ',');
    qpk = find(strcmp(FKEYS,'q_peak_sample'),1);
    rpk = find(strcmp(FKEYS,'r_peak_sample'),1);
    spk = find(strcmp(FKEYS,'s_peak_sample'),1);

    % index worklist rows by unit key (used to materialise freshly-edited rows)
    keyOfRow = @(i) char(T.record_id(i) + "|" + T.lead(i) + "|" + string(T.beat_id(i)));
    rowIndex = containers.Map('KeyType','char','ValueType','double');
    for i = 1:height(T); rowIndex(keyOfRow(i)) = i; end

    % build a fresh CSV line for every touched unit that exists in the worklist
    fresh = containers.Map('KeyType','char','ValueType','char');
    ks = keys(corr);
    for a = 1:numel(ks)
        key = ks{a}; c = corr(key);
        if ~isKey(rowIndex,key); continue; end
        i = rowIndex(key);
        fv = c.fv;
        pp = double(any(~isnan(fv(PGRP))));
        qp = double(any(~isnan(fv(QGRP))));
        tp = double(any(~isnan(fv(TGRP))));
        qq = double(~isnan(fv(qpk)));
        rr = double(~isnan(fv(rpk)));
        ss = double(~isnan(fv(spk)));
        vals = cell(1,numel(cols));
        for j = 1:numel(cols)
            col = cols{j};
            switch col
                case FKEYS
                    kk = find(strcmp(FKEYS,col),1);
                    if isnan(fv(kk)); vals{j} = ''; else; vals{j} = num2str(round(fv(kk))); end
                case 'p_present';   vals{j} = num2str(pp);
                case 'qrs_present'; vals{j} = num2str(qp);
                case 'q_present';   vals{j} = num2str(qq);
                case 'r_present';   vals{j} = num2str(rr);
                case 's_present';   vals{j} = num2str(ss);
                case 't_present';   vals{j} = num2str(tp);
                case 'label_source';vals{j} = 'manual_corrected';
                case 'reviewed';    vals{j} = '1';   % any unit written here was edited, so merge-ready
                case 'edited_at';   vals{j} = c.edited_at;
                otherwise
                    vals{j} = strrep(cstr(T.(col)(i)), ',', ';');
            end
        end
        fresh(key) = strjoin(vals, ',');
    end

    % union: preserve every row already on disk verbatim, replacing only the
    % units edited this session, then append any freshly-edited new units.
    ordered = {}; seen = containers.Map('KeyType','char','ValueType','logical');
    if exist(outCsv,'file') == 2
        try; txt = fileread(outCsv); catch; txt = ''; end
        rawlines = regexp(txt, '\r\n|\n|\r', 'split');
        for li = 1:numel(rawlines)
            ln = rawlines{li};
            if isempty(strtrim(ln)); continue; end
            if startsWith(ln, 'record_id,'); continue; end
            parts = strsplit(ln, ',', 'CollapseDelimiters', false);
            if numel(parts) < 4; continue; end
            k = char(string(parts{1}) + "|" + string(parts{3}) + "|" + string(parts{4}));
            if isKey(seen,k); continue; end
            if isKey(fresh,k)
                ordered{end+1} = fresh(k); %#ok<AGROW>
            else
                ordered{end+1} = migrateLine(parts, cols, FKEYS); %#ok<AGROW>
            end
            seen(k) = true;
        end
    end
    fk = keys(fresh);
    for a = 1:numel(fk)
        if ~isKey(seen, fk{a}); ordered{end+1} = fresh(fk{a}); seen(fk{a}) = true; end %#ok<AGROW>
    end

    % atomic write via a temp file, keeping one rolling backup of the prior file
    if exist(outCsv,'file') == 2
        try; copyfile(outCsv, [outCsv '.autosave.bak']); catch; end
    end
    tmp = [outCsv '.tmp'];
    fid = fopen(tmp,'w');
    if fid < 0; warning('Cannot open %s for writing.', tmp); return; end
    fprintf(fid,'%s\n', headerLine);
    for a = 1:numel(ordered); fprintf(fid,'%s\n', ordered{a}); end
    fclose(fid);
    ok = true;
    try; movefile(tmp, outCsv, 'f'); catch; ok = false; end
    if ~ok; warning('Atomic replace failed; your edits are safe in %s', tmp); end
end

function ln = migrateLine(parts, cols, FKEYS)
%MIGRATELINE  Bring a corrections row written before the q, r and s presence
%   columns existed up to the current schema. The three values are recovered
%   from the row's own q, r and s peak sample fields, where an empty field
%   already meant that landmark was absent, so nothing is invented and no
%   earlier manual decision is overwritten. A row of any other width is passed
%   through untouched.
    ln = strjoin(parts, ',');
    if numel(parts) ~= numel(cols) - 3; return; end
    base = 6;                                   % identity columns before FKEYS
    idx  = base + [find(strcmp(FKEYS,'q_peak_sample'),1), ...
                   find(strcmp(FKEYS,'r_peak_sample'),1), ...
                   find(strcmp(FKEYS,'s_peak_sample'),1)];
    add = cell(1,3);
    for a = 1:3
        add{a} = num2str(double(~isempty(strtrim(parts{idx(a)}))));
    end
    at = find(strcmp(cols,'q_present'),1) - 1;  % old fields that precede the insert
    ln = strjoin([parts(1:at), add, parts(at+1:end)], ',');
end
