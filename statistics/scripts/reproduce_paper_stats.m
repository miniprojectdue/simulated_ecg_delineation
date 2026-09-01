%% reproduce_paper_stats.m

clear; clc;
SCRIPT_DIR   = fileparts(mfilename('fullpath'));
DATASET_ROOT = fileparts(SCRIPT_DIR);
FID = fullfile(SCRIPT_DIR,'medalcare_fiducials_ecgdeli.csv');
MAN = fullfile(SCRIPT_DIR,'medalcare_manifest.csv');
FS  = 500;  MS = 1000/FS;                 % 2 ms per sample
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];
N_AMP_REC = 200;                          % records sampled for amplitude stats

% ---- Table 6 "sim" (healthy) means per lead [Pdur QRSdur Tdur PQint QTint RRint] ----
T6mu = [124.06 131.31 178.12 128.07 310.54 758.15;   % I
        128.09 126.10 182.33 127.18 317.08 758.02;   % II
        164.52 126.80 183.16 171.88 306.94 757.99;   % III
        127.42 128.81 179.38 126.19 318.73 757.97;   % aVR
        154.50 128.62 182.76 169.37 299.19 758.08;   % aVL
        141.00 125.02 184.05 142.60 310.43 758.06;   % aVF
        140.78 129.05 180.72 160.57 303.89 758.06;   % V1
        155.93 136.12 176.15 181.65 287.71 757.90;   % V2
        154.23 132.80 179.01 174.20 285.50 758.01;   % V3
        140.60 127.32 180.05 148.55 290.09 758.03;   % V4
        128.69 123.89 177.39 126.73 310.74 758.12;   % V5
        123.44 126.55 174.63 118.63 320.51 758.06];  % V6
TIMING = ["Pdur","QRSdur","Tdur","PQint","QTint","RRint"];

%% ---------------- 1. Load labels and derive timing features -----------------
T = readtable(FID);
T = sortrows(T, {'record_id','lead','beat_id'});

Pdur   = (T.p_offset_sample  - T.p_onset_sample ) * MS;
QRSdur = (T.qrs_offset_sample- T.qrs_onset_sample) * MS;
Tdur   = (T.t_offset_sample  - T.t_onset_sample ) * MS;
PQint  = (T.qrs_onset_sample - T.p_onset_sample ) * MS;
QTint  = (T.t_offset_sample  - T.qrs_onset_sample) * MS;

% RR interval consecutive R-peak difference within each (record, lead)
g  = findgroups(string(T.record_id), string(T.lead));
RR = splitapply(@(x){[NaN; diff(x)]}, T.r_peak_sample, g);
RRint = vertcat(RR{:}) * MS;

F = table(string(T.record_id), string(T.lead), string(T.disease_class), ...
    Pdur,QRSdur,Tdur,PQint,QTint,RRint, ...
    'VariableNames',[{'record_id','lead','disease_class'}, cellstr(TIMING)]);
for k = 1:numel(TIMING)                    % drop non-physiological (<=0)
    v = F.(TIMING(k)); v(v<=0) = NaN; F.(TIMING(k)) = v;
end

% ---- aggregate to ONE value per record-lead (median across beats) ----
R = groupsummary(F, {'record_id','lead','disease_class'}, 'median', cellstr(TIMING));
for k = 1:numel(TIMING), R.(TIMING(k)) = R.("median_"+TIMING(k)); end

%% ---------------- 2. Healthy timing vs Table 6 (sim) ------------------------
fprintf('\n=== TIMING: sinus per-record vs Table 6 (sim), per lead ===\n');
for f = 1:numel(TIMING)
    dmu = zeros(numel(LEADS),1); sr = zeros(numel(LEADS),1);
    for L = 1:numel(LEADS)
        v = R.(TIMING(f))(R.disease_class=="sinus" & R.lead==LEADS(L));
        dmu(L) = mean(v,'omitnan') - T6mu(L,f);
        sr(L)  = std(v,'omitnan');
    end
    fprintf('%-8s meanDeltaMu %+6.1f ms   ourSigma %5.1f ms\n', TIMING(f), mean(dmu), mean(sr));
end

%% ---------------- 3. Per-class shift vs sinus (Fig. 6) ----------------------
classes = ["sinus","avblock","lbbb","rbbb","iab","lae","fam","mi"];
base = zeros(1,numel(TIMING));
for f = 1:numel(TIMING), base(f) = mean(R.(TIMING(f))(R.disease_class=="sinus"),'omitnan'); end
fprintf('\n=== PER-CLASS mean shift vs sinus (ms, per-record, averaged over leads) ===\n');
for c = classes
    d = zeros(1,numel(TIMING));
    for f = 1:numel(TIMING)
        d(f) = mean(R.(TIMING(f))(R.disease_class==c),'omitnan') - (c~="sinus")*base(f);
    end
    fprintf('%-9s Pdur %+6.1f  QRS %+6.1f  PQ %+6.1f  QT %+6.1f\n', c, d(1),d(2),d(4),d(5));
end

%% ---------------- 4. Amplitude features (sampled, per record) --------------
Mn = readtable(MAN,'TextType','string');
sinRec = unique(T.record_id(string(T.disease_class)=="sinus"));
sinRec = string(sinRec(1:min(N_AMP_REC,numel(sinRec))));
pk = ["p_peak_sample","q_peak_sample","r_peak_sample","s_peak_sample","t_peak_sample"];
ampName = ["Pamp","Qamp","Ramp","Samp","Tamp"];
ampRec = struct('Pamp',[],'Qamp',[],'Ramp',[],'Samp',[],'Tamp',[]);
for i = 1:numel(sinRec)
    rid = sinRec(i);
    mrow = Mn(Mn.record_id==rid,:); if isempty(mrow), continue; end
    sig = readmatrix(fullfile(DATASET_ROOT, mrow.path_raw(1)));
    if size(sig,1)==12, sig = sig.'; end
    sII = sig(:, LEADS=="II");
    rows = T(T.record_id==rid & string(T.lead)=="II", :);
    tmp = struct('Pamp',[],'Qamp',[],'Ramp',[],'Samp',[],'Tamp',[]);
    for b = 1:height(rows)
        qon = rows.qrs_onset_sample(b); if isnan(qon), continue; end
        w0 = max(1, qon-11);  base_mV = median(sII(w0:qon+1));
        for a = 1:numel(pk)
            s = rows.(pk(a))(b); if isnan(s), continue; end
            idx = s + 1;
            if idx>=1 && idx<=numel(sII), tmp.(ampName(a))(end+1) = sII(idx)-base_mV; end
        end
    end
    for a = 1:numel(ampName)             % one median per record
        if ~isempty(tmp.(ampName(a))), ampRec.(ampName(a))(end+1) = median(tmp.(ampName(a))); end
    end
end

%% ---------------- 5. Gaussian overlays (Fig. 5 style, lead II) --------------
T6sd = [18.37 17.56 31.08 28.06 33.23 54.97;  14.00 13.73 25.94 22.97 23.61 54.41];
titles = ["P duration","QRS duration","T duration","PQ interval","QT interval","RR interval"];
figure('Position',[100 100 1400 700]);
for f = 1:6
    subplot(2,3,f);
    d = R.(TIMING(f))(R.disease_class=="sinus" & R.lead=="II"); d = d(~isnan(d) & d>0);
    histogram(d,'Normalization','pdf','FaceColor',[0.30 0.47 0.66],'EdgeColor','none','FaceAlpha',0.55);
    hold on;
    mu = T6mu(2,f); sd = T6sd(2,f);
    xs = linspace(min(d), max(d), 300);
    plot(xs, exp(-0.5*((xs-mu)/sd).^2)/(sd*sqrt(2*pi)), 'r-', 'LineWidth', 2);
    xline(mean(d),'--','Color',[0.30 0.47 0.66]); xline(mu,'r--');
    title(titles(f)+" (lead II)"); xlabel('ms'); set(gca,'YTick',[]);
    if f==1, legend({'Our labels (per record)','Paper Table 6'},'Location','northwest'); end
end
sgtitle('Healthy sinus timing (one value per record): our ECGdeli labels vs MedalCare-XL Table 6');
exportgraphics(gcf, fullfile(SCRIPT_DIR,'reproduce_timing_matlab.png'), 'Resolution', 130);
fprintf('\nSaved reproduce_timing_matlab.png\n');
