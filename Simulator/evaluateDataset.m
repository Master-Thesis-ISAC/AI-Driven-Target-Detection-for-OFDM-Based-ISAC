function results = evaluateDataset(outputDir)
% evaluateDataset  Dataset sanity check and classical detector metrics.
%
% Loads manifest.csv, computes per-SNR Pd/Pfa for CA-CFAR and energy
% detector, plots BER vs SNR_comm, and writes figures to outputDir/figs/.

if nargin < 1 || isempty(outputDir); outputDir = 'output'; end
manifestFile = fullfile(outputDir, 'manifest.csv');
if ~exist(manifestFile,'file')
    error('evaluateDataset:noManifest', 'No manifest.csv in %s', outputDir);
end

T = readtable(manifestFile);
fprintf('Loaded %d scenarios from %s\n', height(T), manifestFile);

figDir = fullfile(outputDir, 'figs');
if ~exist(figDir,'dir'); mkdir(figDir); end

% Dataset balance
n_present = sum(T.targetPresent == 1);
n_absent  = sum(T.targetPresent == 0);
fprintf('Target-present: %d  |  Target-absent: %d  (%.1f%% / %.1f%%)\n', ...
    n_present, n_absent, 100*n_present/height(T), 100*n_absent/height(T));

if any(strcmp('split', T.Properties.VariableNames))
    fprintf('Split sizes:  train=%d  val=%d  test=%d\n', ...
        sum(strcmp(T.split,'train')), ...
        sum(strcmp(T.split,'val')), ...
        sum(strcmp(T.split,'test')));
end
disp(groupcounts(T,'type'));

% Pd/Pfa vs sensing SNR
SNR_grid = unique(T.SNR_sense_dB);
nSNR     = numel(SNR_grid);
Pd_CFAR  = zeros(nSNR,1); Pd_Ene  = zeros(nSNR,1);
Pfa_CFAR = zeros(nSNR,1); Pfa_Ene = zeros(nSNR,1);
n_pres   = zeros(nSNR,1); n_abs   = zeros(nSNR,1);
hasSize  = any(strcmp('sceneLargestSize', T.Properties.VariableNames));
Pd_CFAR_size = nan(nSNR,3);

for i = 1:nSNR
    snr_i = SNR_grid(i);
    pres  = T(T.SNR_sense_dB==snr_i & T.targetPresent==1,:);
    abs_  = T(T.SNR_sense_dB==snr_i & T.targetPresent==0,:);
    n_pres(i) = height(pres); n_abs(i) = height(abs_);
    if height(pres)>0; Pd_CFAR(i)=mean(pres.CFAR_present); Pd_Ene(i)=mean(pres.Energy_present); end
    if height(abs_)>0; Pfa_CFAR(i)=mean(abs_.CFAR_present); Pfa_Ene(i)=mean(abs_.Energy_present); end
    if hasSize
        for sz = 1:3
            sub = pres(pres.sceneLargestSize==sz,:);
            if height(sub)>0; Pd_CFAR_size(i,sz)=mean(sub.CFAR_present); end
        end
    end
end

fig = figure('Name','Classical detection vs SNR','Color','w','Position',[80 80 1300 420]);
subplot(1,3,1); hold on; grid on; box on;
plot(SNR_grid,Pd_CFAR,'o-','LineWidth',2,'DisplayName','CA-CFAR P_d');
plot(SNR_grid,Pd_Ene, 's-','LineWidth',2,'DisplayName','Energy P_d');
plot(SNR_grid,Pfa_CFAR,'o--','LineWidth',1.2,'DisplayName','CA-CFAR P_{fa}');
plot(SNR_grid,Pfa_Ene, 's--','LineWidth',1.2,'DisplayName','Energy P_{fa}');
xlabel('SNR_{sense} (dB)'); ylabel('Probability'); ylim([0 1.05]);
title('Classical detection (all targets)'); legend('Location','east','FontSize',8);

subplot(1,3,2); hold on; grid on; box on;
if hasSize
    plot(SNR_grid,Pd_CFAR_size(:,1),'g-o','LineWidth',2,'DisplayName','small');
    plot(SNR_grid,Pd_CFAR_size(:,2),'b-s','LineWidth',2,'DisplayName','medium');
    plot(SNR_grid,Pd_CFAR_size(:,3),'r-^','LineWidth',2,'DisplayName','large');
    xlabel('SNR_{sense} (dB)'); ylabel('P_d (CFAR)'); ylim([0 1.05]);
    title('CFAR P_d by target size'); legend('Location','southeast','FontSize',8);
else
    text(0.5,0.5,'sceneLargestSize not in manifest','HorizontalAlignment','center');
end

subplot(1,3,3); hold on; grid on; box on;
bar(SNR_grid,[n_pres n_abs],'stacked');
xlabel('SNR_{sense} (dB)'); ylabel('Scenario count');
title('Dataset balance per SNR bin');
legend({'target-present','target-absent'},'Location','northwest','FontSize',8);

saveas(fig, fullfile(figDir,'RQ2_classical_detection.png'));
fprintf('Saved %s\n', fullfile(figDir,'RQ2_classical_detection.png'));

% CFAR ROC sweep from saved Pfa grids
nSample       = min(1000, height(T));
sampleIdx     = randperm(height(T), nSample);
PfaGrid_master = []; Pd_acc=[]; Pfa_acc=[]; nP=0; nA=0;

for ii = 1:nSample
    f = T.file{sampleIdx(ii)};
    if ~exist(f,'file'); continue; end
    S = load(f,'classicalCFAR_Pfa_grid','classicalCFAR_grid','targetPresent');
    if ~isfield(S,'classicalCFAR_Pfa_grid')||isempty(S.classicalCFAR_Pfa_grid); continue; end
    if isempty(PfaGrid_master)
        PfaGrid_master = S.classicalCFAR_Pfa_grid(:);
        Pd_acc  = zeros(numel(PfaGrid_master),1);
        Pfa_acc = zeros(numel(PfaGrid_master),1);
    end
    g = S.classicalCFAR_grid(:);
    if numel(g)~=numel(PfaGrid_master); continue; end
    if S.targetPresent; Pd_acc=Pd_acc+double(g); nP=nP+1;
    else;               Pfa_acc=Pfa_acc+double(g); nA=nA+1; end
end

if nP>0 && nA>0
    Pd_meas  = Pd_acc/nP;
    Pfa_meas = Pfa_acc/nA;
    fig = figure('Name','CFAR ROC operating points','Color','w','Position',[80 80 700 520]);
    hold on; grid on; box on;
    plot([0 1],[0 1],'k--','LineWidth',1,'DisplayName','random');
    plot(Pfa_meas,Pd_meas,'bo','LineWidth',2,'MarkerSize',12,'MarkerFaceColor','b','DisplayName','CA-CFAR');
    for k = 1:numel(PfaGrid_master)
        text(Pfa_meas(k)+0.015, Pd_meas(k), ...
             sprintf('P_{fa}^{set}=10^{%d}',round(log10(PfaGrid_master(k)))),'FontSize',8,'Color','b');
    end
    xlim([-0.02 1.02]); ylim([0 1.05]);
    xlabel('Measured P_{fa}'); ylabel('Measured P_d');
    title(sprintf('CFAR ROC (%d scenarios)', nP+nA)); legend('Location','southeast');
    saveas(fig, fullfile(figDir,'RQ2_CFAR_ROC.png'));
    fprintf('Saved %s\n', fullfile(figDir,'RQ2_CFAR_ROC.png'));
end

% BER vs SNR_comm
SNRc_grid = unique(T.SNR_comm_dB);
nSNRc     = numel(SNRc_grid);
BER_med=zeros(nSNRc,1); BER_q1=zeros(nSNRc,1); BER_q3=zeros(nSNRc,1); EVM_med=zeros(nSNRc,1);
for i = 1:nSNRc
    rows = T(T.SNR_comm_dB==SNRc_grid(i),:);
    ber  = rows.BER(~isnan(rows.BER));
    evm  = rows.EVM_dB(~isnan(rows.EVM_dB));
    if isempty(ber); continue; end
    BER_med(i)=median(ber); BER_q1(i)=quantile(ber,0.25); BER_q3(i)=quantile(ber,0.75);
    EVM_med(i)=median(evm);
end

fig = figure('Name','BER and EVM vs SNR_comm','Color','w','Position',[80 80 1100 420]);
subplot(1,2,1);
errorbar(SNRc_grid,BER_med,BER_med-BER_q1,BER_q3-BER_med,'o-','LineWidth',2);
set(gca,'YScale','log'); grid on; box on;
xlabel('SNR_{comm} (dB)'); ylabel('BER (uncoded)'); title('BER vs SNR_{comm}');

subplot(1,2,2);
plot(SNRc_grid,EVM_med,'s-','LineWidth',2);
grid on; box on;
xlabel('SNR_{comm} (dB)'); ylabel('Median EVM (dB)'); title('EVM vs SNR_{comm}');

saveas(fig, fullfile(figDir,'RQ3_BER_vs_SNR.png'));
fprintf('Saved %s\n', fullfile(figDir,'RQ3_BER_vs_SNR.png'));

% Dataset health histograms
fig = figure('Name','Dataset health check','Color','w','Position',[80 80 1100 420]);
subplot(1,3,1); histogram(T.SNR_sense_dB); grid on; title('SNR_{sense}'); xlabel('dB');
subplot(1,3,2); histogram(T.SNR_comm_dB);  grid on; title('SNR_{comm}');  xlabel('dB');
subplot(1,3,3); histogram(T.numObj);        grid on; title('Targets per scenario'); xlabel('count');
saveas(fig, fullfile(figDir,'dataset_health.png'));
fprintf('Saved %s\n', fullfile(figDir,'dataset_health.png'));

results = struct( ...
    'numScenarios',     height(T), ...
    'numTargetPresent', n_present, ...
    'numTargetAbsent',  n_absent, ...
    'SNR_sense_grid',   SNR_grid, ...
    'Pd_CFAR',          Pd_CFAR, ...
    'Pd_Energy',        Pd_Ene, ...
    'Pfa_CFAR',         Pfa_CFAR, ...
    'Pfa_Energy',       Pfa_Ene, ...
    'SNR_comm_grid',    SNRc_grid, ...
    'BER_median',       BER_med, ...
    'BER_q1',           BER_q1, ...
    'BER_q3',           BER_q3, ...
    'EVM_median',       EVM_med);

fprintf('\nTotal: %d scenarios  (present:%d  absent:%d)\n', height(T), n_present, n_absent);
fprintf('CFAR P_d (high-SNR): %.3f  P_fa: %.3f\n', Pd_CFAR(end), Pfa_CFAR(end));
fprintf('Median BER (high SNR_comm): %.2e\n', BER_med(end));
fprintf('Figures written to: %s\n', figDir);
end
