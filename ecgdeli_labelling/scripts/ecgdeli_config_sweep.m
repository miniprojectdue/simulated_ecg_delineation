%% ecgdeli_config_sweep.m
% Finds the ECGdeli preprocessing config that best reproduces the paper's Table 6.
%
% Runs a small HEALTHY (sinus) sample under four preprocessing configs and writes
% one minimal fiducials CSV per config. Then score each in Python
%     python3 score_vs_table6.py sweep_raw_iso.csv
%     python3 score_vs_table6.py sweep_filtered_iso.csv   ... etc.
% Keep the config with the smallest QT / total distance-to-paper, then set that
% SIGNAL_VER + APPLY_ISOLINE in run_ecgdeli_medalcare.m and relabel the full set.
%
% Requires ECGdeli on the path (same as the main driver).

clear; clc;
SCRIPT_DIR   = fileparts(mfilename('fullpath'));            % ecgdeli_labelling/scripts
REPO_ROOT    = fileparts(fileparts(SCRIPT_DIR));            % repo root (holds the signal files)
DATASET_ROOT = REPO_ROOT;
ECGDELI_PATH = fullfile(REPO_ROOT,'ECG_TOOL','ECGdeli');
MANIFEST     = fullfile(REPO_ROOT,'ecgdeli_labelling','data','input','medalcare_manifest.csv');
OUTDIR       = fullfile(REPO_ROOT,'ecgdeli_labelling','logs');   % diagnostic sweep_*.csv scratch
NREC = 30;          % sinus records per config (enough for stable per-lead means)
FS   = 500;

addpath(genpath(ECGDELI_PATH));
assert(exist('Annotate_ECG_Multi','file')==2,'ECGdeli not on path.');

T = readtable(MANIFEST,'Delimiter',',','TextType','string');
for vn = string(T.Properties.VariableNames)
    if isstring(T.(vn)), T.(vn) = fillmissing(T.(vn),'constant',""); end
end
sin = T(T.disease_class=="sinus",:);
sin = sin(1:min(NREC,height(sin)),:);
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];

% name , signal-version column , apply isoline?
configs = { 'raw_iso',       'path_raw',      true ; ...
            'raw_noiso',     'path_raw',      false; ...
            'filtered_iso',  'path_filtered', true ; ...
            'filtered_noiso','path_filtered', false };

for ci = 1:size(configs,1)
    name = configs{ci,1}; vercol = configs{ci,2}; useiso = configs{ci,3};
    outcsv = fullfile(OUTDIR, "sweep_"+name+".csv");
    fid = fopen(outcsv,'w');
    fprintf(fid,['record_id,disease_class,lead,beat_id,p_onset_sample,p_offset_sample,' ...
                 'qrs_onset_sample,qrs_offset_sample,t_onset_sample,t_offset_sample,r_peak_sample\n']);
    for k = 1:height(sin)
        rid = sin.record_id(k);
        fp  = fullfile(DATASET_ROOT, sin.(char(vercol))(k));
        try
            M = readmatrix(fp); if size(M,1)==12, sig = M.'; else, sig = M; end
            sig = double(sig);
            if useiso, try sig = Isoline_Correction(sig); catch; end; end
            [~, FPT_Cell] = Annotate_ECG_Multi(sig, FS, 'PQRST');
            if isempty(FPT_Cell), continue; end
            for L = 1:numel(LEADS)
                f = FPT_Cell{L}; if isempty(f), continue; end
                for b = 1:size(f,1)
                    g = @(c) col0(f,b,c);
                    fprintf(fid,'%s,sinus,%s,%d,%s,%s,%s,%s,%s,%s,%s\n', rid, LEADS(L), b, ...
                        g(1),g(3),g(4),g(8),g(10),g(12),g(6));   % Pon,Poff,QRSon,QRSoff,Ton,Toff,R
                end
            end
        catch ME
            fprintf('  %s (%s): %s\n', rid, name, ME.message);
        end
    end
    fclose(fid);
    fprintf('wrote %s\n', outcsv);
end
fprintf(['\nScore each config:\n' ...
         '  python3 score_vs_table6.py sweep_raw_iso.csv\n' ...
         '  python3 score_vs_table6.py sweep_raw_noiso.csv\n' ...
         '  python3 score_vs_table6.py sweep_filtered_iso.csv\n' ...
         '  python3 score_vs_table6.py sweep_filtered_noiso.csv\n' ...
         'Pick the smallest QT / total distance, then set SIGNAL_VER + APPLY_ISOLINE\n' ...
         'in run_ecgdeli_medalcare.m to match and relabel the full dataset.\n']);

function s = col0(f,b,c)   % FPT value -> 0-based string, "" if missing
    if c > size(f,2), s = ""; return; end
    x = f(b,c);
    if isnan(x) || x <= 0, s = ""; else, s = string(round(x-1)); end
end
