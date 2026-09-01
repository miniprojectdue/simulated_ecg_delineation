%% run_ecgdeli_medalcare.m
% Auto-delineates the MedalCare-XL dataset with ECGdeli and writes fiducials
% in the project "fiducials_per_lead" schema, with the disease class preserved.
%
%   ECGdeli takes an in-memory matrix, so we do NOT need to duplicate 27 GB this script reads
%   each original CSV, transposes it (the files are leads x samples, ECGdeli
%   requires samples x leads), runs the delineator, and appends one row per
%   beat-per-lead to a master CSV.
%
% REQUIREMENTS - if you want to run on your end
%   - MATLAB with image, signal, statistics, wavelet toolboxes
%   - ECGdeli toolbox https://github.com/KIT-IBT/ECGdeli   (clone it)
%
%
% FPT (ECGdeli fiducial point table) column map used below
%   1=P-onset 2=P-peak 3=P-offset  4=QRS-onset 5=Q-peak 6=R-peak 7=S-peak
%   8=QRS-offset  10=T-onset 11=T-peak 12=T-offset   (col 6 = R confirmed in
%   Annotate_ECG_Multi, cols 1,3,4,12 confirmed in the MedalCare synthesis code)

clear; clc;

%% ----------------------------- CONFIG -----------------------------------
% Layout this script lives at ecgdeli_labelling/scripts/. The repo root (which contains
% WP2_largeDataset_Noise and ECG_TOOL/ECGdeli) is two levels up. 
SCRIPT_DIR    = fileparts(mfilename('fullpath'));                 % ecgdeli_labelling/scripts
REPO_ROOT     = fileparts(fileparts(SCRIPT_DIR));                 % repo root (holds the signal files)
DATASET_ROOT  = REPO_ROOT;                                        % raw signal path_raw is relative to here
ECGDELI_PATH  = fullfile(REPO_ROOT,'ECG_TOOL','ECGdeli');         % the ECGdeli clone
MANIFEST      = fullfile(REPO_ROOT,'ecgdeli_labelling','data','input','medalcare_manifest.csv');
OUTPUT_CSV    = fullfile(REPO_ROOT,'ecgdeli_labelling','data','primary','medalcare_fiducials_ecgdeli.csv');
FAIL_LOG      = fullfile(REPO_ROOT,'ecgdeli_labelling','logs','ecgdeli_failures.log');
SIGNAL_VER    = 'raw';        % 'raw' (clean) | 'filtered' | 'noise'
APPLY_ISOLINE = true;         % ECGdeli isoline correction before annotation
FS            = 500;
START_IDX     = 1;            % process manifest rows START_IDXEND_IDX
END_IDX       = Inf;           % set to Inf for the full run after the smoke test
%% ------------------------------------------------------------------------

addpath(genpath(ECGDELI_PATH));
assert(exist('Annotate_ECG_Multi','file')==2, ...
    'ECGdeli not on path. Set ECGDELI_PATH to the ECGdeli clone.');

T = readtable(MANIFEST,'Delimiter',',','TextType','string');
% empty string cells (e.g. mi_subclass for non-MI classes) read as <missing>
% convert to "" so strjoin() works downstream
for vn = string(T.Properties.VariableNames)
    if isstring(T.(vn)), T.(vn) = fillmissing(T.(vn),'constant',""); end
end
END_IDX = min(END_IDX, height(T));
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];
pathCol = char("path_" + SIGNAL_VER);

% ---- resume skip records already present in the output ----
done = strings(0,1); writeHeader = true;
if isfile(OUTPUT_CSV)
    try
        E = readtable(OUTPUT_CSV,'Delimiter',',','TextType','string');
        writeHeader = false;
        if any(strcmp('record_id',E.Properties.VariableNames)) && height(E) > 0
            done = unique(string(E.record_id));   % force string so done==rid is valid
        end
    catch
    end
end

HEADER = strjoin(["record_id","source","split","disease_class","mi_subclass", ...
 "fs_hz","n_samples","duration_sec","lead","beat_id", ...
 "beat_start_sample","beat_end_sample","p_onset_sample","p_peak_sample","p_offset_sample", ...
 "qrs_onset_sample","q_peak_sample","r_peak_sample","s_peak_sample","qrs_offset_sample", ...
 "t_onset_sample","t_peak_sample","t_offset_sample","u_onset_sample","u_peak_sample","u_offset_sample", ...
 "p_onset_ms","p_peak_ms","p_offset_ms","qrs_onset_ms","q_peak_ms","r_peak_ms","s_peak_ms","qrs_offset_ms", ...
 "t_onset_ms","t_peak_ms","t_offset_ms","u_onset_ms","u_peak_ms","u_offset_ms", ...
 "p_present","qrs_present","t_present","u_present","label_source","label_quality","notes", ...
 "qt_interval_ms","qt_peak_ms", ...
 "ecgdeli_pdur_ms","ecgdeli_qrsdur_ms","ecgdeli_tdur_ms","ecgdeli_pq_ms","ecgdeli_pr_ms","ecgdeli_qt_ms","ecgdeli_rr_ms", ...
 "ecgdeli_sync_pdur_ms","ecgdeli_sync_qrsdur_ms","ecgdeli_sync_tdur_ms","ecgdeli_sync_pq_ms","ecgdeli_sync_pr_ms","ecgdeli_sync_qt_ms","ecgdeli_sync_qtc_ms","ecgdeli_sync_rr_ms"], ',');

fid = fopen(OUTPUT_CSV,'a');
if writeHeader, fprintf(fid,'%s\n',HEADER); end
flog = fopen(FAIL_LOG,'a');
nOK = 0; nFail = 0;

for k = START_IDX:END_IDX
    rid = T.record_id(k);
    if any(done==rid), continue; end
    fpath = fullfile(DATASET_ROOT, T.(pathCol)(k));
    try
        M = readmatrix(fpath);                    % 12 x 5000 (leads x samples)
        if size(M,1)==12 && size(M,2)~=12, sig = M.'; else, sig = M; end  % -> samples x leads
        sig = double(sig);
        n = size(sig,1);

        if APPLY_ISOLINE
            try
                sig = Isoline_Correction(sig);
            catch
                if k==START_IDX, warning('Isoline_Correction unavailable - proceeding without it.'); end
            end
        end

        [FPT_MultiChannel, FPT_Cell] = Annotate_ECG_Multi(sig, FS, 'PQRST');
        if isempty(FPT_Cell) || all(cellfun(@isempty,FPT_Cell))
            fprintf(flog,'%s\tempty_FPT\n',rid); nFail=nFail+1; continue;
        end

        % ECGdeli's own interval features, computed once from this record's FPT
        %   features_leadwise (L x B x 7) 1 Pdur 2 QRSdur 3 Tdur 4 PQ 5 PR 6 QT 7 RR
        %   features_sync     (B x 8, cross-lead) 1 Pdur 2 QRSdur 3 Tdur 4 PQ 5 PR
        %                                          6 QT 7 QTc(Framingham) 8 RR
        % We write BOTH sets to the CSV (columns ecgdeli_* and ecgdeli_sync_*). The
        % lead-wise QT (feature 6) also populates qt_interval_ms. If it is
        % unavailable or errors (e.g. leads with unequal beat counts) the lead-wise
        % columns fall back to our identical inline subtraction (verified 0.000 ms in
        % verify_ecgdeli_qt.m), the cross-lead sync columns are ECGdeli-only.
        try
            [features_leadwise, features_sync] = ExtractIntervalFeaturesFromFPT(FPT_Cell, FPT_MultiChannel);
        catch
            features_leadwise = []; features_sync = [];
        end

        disease = T.disease_class(k); subc = T.mi_subclass(k); split = T.split(k);
        buf = strings(0,1);
        for L = 1:numel(LEADS)
            fpt = FPT_Cell{L};
            if isempty(fpt), continue; end
            Rc = fpt(:,6);
            for b = 1:size(fpt,1)
                % Fiducials are read STRAIGHT from ECGdeli's FPT columns -- this script adds
                % no search window, buffer, or offset of its own. QT is later T-offset minus
                % QRS-onset, i.e. column 12 minus column 4, so if the QT looked wrong these two
                % column indices (and ECGdeli's own T-end detection) are the things we need to check.
                Pon=cs(fpt,b,1); Ppk=cs(fpt,b,2); Poff=cs(fpt,b,3);
                QRSon=cs(fpt,b,4); Qpk=cs(fpt,b,5); Rpk=cs(fpt,b,6); Spk=cs(fpt,b,7); QRSoff=cs(fpt,b,8);   % col 4 = QRS-onset
                Ton=cs(fpt,b,10); Tpk=cs(fpt,b,11); Toff=cs(fpt,b,12);   % col 12 = T-offset
                if b==1, bs=0; else, bs=max(0,floor((Rc(b-1)+Rc(b))/2)-1); end
                if b==size(fpt,1), be=n-1; else, be=floor((Rc(b)+Rc(b+1))/2)-1; end
                % --- ECGdeli lead-wise interval features (ms) ---------------------
                % intv() uses the ECGdeli value when available, else the identical
                % inline subtraction, returns "" if either constituent fiducial is
                % absent, so undetected waves give empty cells (no spurious values).
                ed_pdur = intv(lw(features_leadwise,L,b,1), Poff, Pon,   FS);   % P duration
                ed_qrsd = intv(lw(features_leadwise,L,b,2), QRSoff,QRSon,FS);   % QRS duration
                ed_tdur = intv(lw(features_leadwise,L,b,3), Toff, Ton,   FS);   % T duration
                ed_pq   = intv(lw(features_leadwise,L,b,4), QRSon,Pon,   FS);   % PQ interval
                ed_pr   = intv(lw(features_leadwise,L,b,5), Rpk,  Pon,   FS);   % PR interval
                ed_qt   = intv(lw(features_leadwise,L,b,6), Toff, QRSon, FS);   % QT interval
                ed_rr   = lw(features_leadwise,L,b,7);                          % RR (ECGdeli)
                if strlength(ed_rr)==0                                          % RR fallback
                    if b < size(fpt,1),  ed_rr = string(round((Rc(b+1)-Rc(b))*1000/FS,3));
                    elseif b > 1,        ed_rr = string(round((Rc(b)-Rc(b-1))*1000/FS,3)); end
                end
                % --- ECGdeli synchronized (cross-lead) features (ms) -------------
                % Same value for every lead of a given beat, ECGdeli-only (no inline
                % fallback because they combine all 12 leads via its maxk/mink rule).
                es_pdur = sy(features_sync,b,1); es_qrsd = sy(features_sync,b,2);
                es_tdur = sy(features_sync,b,3); es_pq   = sy(features_sync,b,4);
                es_pr   = sy(features_sync,b,5); es_qt   = sy(features_sync,b,6);
                es_qtc  = sy(features_sync,b,7); es_rr   = sy(features_sync,b,8);
                row = [rid,"medalcare",split,disease,subc, ...
                    string(FS),string(n),string(n/FS),LEADS(L),string(b), ...
                    string(bs),string(be), ss(Pon),ss(Ppk),ss(Poff), ...
                    ss(QRSon),ss(Qpk),ss(Rpk),ss(Spk),ss(QRSoff), ...
                    ss(Ton),ss(Tpk),ss(Toff),"","","", ...
                    ms(Pon,FS),ms(Ppk,FS),ms(Poff,FS),ms(QRSon,FS),ms(Qpk,FS),ms(Rpk,FS),ms(Spk,FS),ms(QRSoff,FS), ...
                    ms(Ton,FS),ms(Tpk,FS),ms(Toff,FS),"","","", ...
                    pres(Ppk),"1",pres(Tpk),"0", ...
                    "ecgdeli_auto","auto_uncorrected","ECGdeli PQRST; 0-based samples; U not detected", ...
                    ed_qt, iv(Tpk,QRSon,FS), ...                       % qt_interval_ms (ECGdeli QT, to T-offset), qt_peak_ms (to T-peak)
                    ed_pdur,ed_qrsd,ed_tdur,ed_pq,ed_pr,ed_qt,ed_rr, ...           % ECGdeli lead-wise set
                    es_pdur,es_qrsd,es_tdur,es_pq,es_pr,es_qt,es_qtc,es_rr];       % ECGdeli synchronized (cross-lead) set
                row(ismissing(row)) = "";        % replace any <missing> with ""
                buf(end+1) = strjoin(row, ',');  %#ok<AGROW> - suppress warning
            end
        end
        if ~isempty(buf)
            fprintf(fid,'%s\n', strjoin(buf, char(10)));
            nOK = nOK + 1;
        end
        if mod(k,100)==0, fprintf('  %d / %d  (ok=%d fail=%d)\n', k, END_IDX, nOK, nFail); end
    catch ME
        fprintf(flog,'%s\t%s\n', rid, ME.message); nFail = nFail + 1;
    end
end
fclose(fid); fclose(flog);
fprintf('DONE. records ok=%d, failed=%d. Output: %s\n', nOK, nFail, OUTPUT_CSV);

%% --------------------------- helpers ------------------------------------
function v = cs(fpt,b,c)          % FPT cell -> 0-based sample, NaN if missing
    % The 1-based->0-based shift (x-1) is applied to EVERY fiducial identically, so it
    % cancels in any interval difference (QT = T_off - QRS_on subtracts two shifted
    % values).This conversion therefore cannot create or affect the QT gap vs the paper.
    if c > size(fpt,2), v = NaN; return; end
    x = fpt(b,c);
    if isnan(x) || x <= 0, v = NaN; else, v = x - 1; end   % MATLAB 1-based -> 0-based
end
function s = ss(v)                % sample -> string ("" if missing)
    if isnan(v), s = ""; else, s = string(round(v)); end
end
function s = ms(v,fs)             % sample -> milliseconds ( = sample * 1000/fs = *2 ms at 500 Hz )
    if isnan(v), s = ""; else, s = string(round(v*1000/fs,3)); end
end
function s = iv(a,b,fs)           % interval (a-b) in ms, "" if either endpoint missing
    % ECGdeli/clinical QT is T-offset - QRS-onset (col 12 - col 4), same as
    % ExtractIntervalFeaturesFromFPT.m in ECGdeli. qt_peak_ms (T-peak - QRS-onset)
    % is a diagnostic that coincides with the paper's Table 6 QT.
    if isnan(a) || isnan(b), s = ""; else, s = string(round((a-b)*1000/fs,3)); end
end
function s = pres(v)              % presence flag
    if isnan(v), s = "0"; else, s = "1"; end
end
function s = lw(F,L,b,c)          % lead-wise ECGdeli feature -> ms string ("" if unavailable/NaN)
    if isempty(F) || L>size(F,1) || b>size(F,2) || c>size(F,3), s=""; return; end
    v = F(L,b,c);
    if isnan(v), s=""; else, s=string(round(v,3)); end
end
function s = sy(F,b,c)            % synchronized (cross-lead) ECGdeli feature -> ms string ("" if unavailable/NaN)
    if isempty(F) || b>size(F,1) || c>size(F,2), s=""; return; end
    v = F(b,c);
    if isnan(v), s=""; else, s=string(round(v,3)); end
end
function s = intv(edStr,a,b,fs)   % prefer ECGdeli value edStr else identical inline (a-b), "" if fiducial missing
    if isnan(a) || isnan(b),        s = "";
    elseif strlength(edStr) > 0,    s = edStr;                              % ECGdeli feature value
    else,                           s = string(round((a-b)*1000/fs,3)); end % identical fallback
end
