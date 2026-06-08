% generate_RQ3_BER_EVM.m
% Reads manifest.csv and plots measured BER and EVM vs communication SNR.
% Produces RQ3_BER_vs_SNR_with_theory.png used in the thesis.

clc; clear; close all;

manifestFile = fullfile('output', 'manifest.csv');
T = readtable(manifestFile);

snrComm_dB = sort(unique(T.SNR_comm_dB));
n          = numel(snrComm_dB);

measuredBER_mean   = zeros(n, 1);
measuredBER_median = zeros(n, 1);
medianEVM_dB       = zeros(n, 1);

for i = 1:n
    idx = T.SNR_comm_dB == snrComm_dB(i);
    measuredBER_mean(i)   = mean(T.BER(idx));
    measuredBER_median(i) = median(T.BER(idx));
    medianEVM_dB(i)       = median(T.EVM_dB(idx));
end

disp(table(snrComm_dB, measuredBER_mean, measuredBER_median, medianEVM_dB, ...
    'VariableNames', {'SNR_comm_dB','Mean_BER','Median_BER','Median_EVM_dB'}));

% Replace zero BER with floor value for log-scale display only
plotBER = measuredBER_mean;
plotBER(plotBER == 0) = 1e-7;

fig = figure('Color','w','Position',[100 100 1400 560]);

% Left panel: BER vs SNR
subplot(1,2,1);
semilogy(snrComm_dB, plotBER, 'o-', 'LineWidth',2.2, 'MarkerSize',8, ...
         'DisplayName','Measured BER');
hold on; grid on; box on;
xlabel('SNR_{comm} (dB)', 'FontSize',12, 'FontWeight','bold');
ylabel('BER (uncoded)',   'FontSize',12, 'FontWeight','bold');
title('Communication BER vs SNR', 'FontSize',13, 'FontWeight','bold');
legend('Location','southwest', 'FontSize',10);
xlim([min(snrComm_dB) max(snrComm_dB)]);
ylim([1e-7 1e-1]);
set(gca,'FontSize',11);

% Right panel: EVM vs SNR
subplot(1,2,2);
plot(snrComm_dB, medianEVM_dB, 's-', 'LineWidth',2.2, 'MarkerSize',8, ...
     'DisplayName','Measured median EVM');
grid on; box on;
xlabel('SNR_{comm} (dB)',  'FontSize',12, 'FontWeight','bold');
ylabel('Median EVM (dB)',  'FontSize',12, 'FontWeight','bold');
title('Communication EVM vs SNR', 'FontSize',13, 'FontWeight','bold');
xlim([min(snrComm_dB) max(snrComm_dB)]);
ylim([min(medianEVM_dB)-2 max(medianEVM_dB)+2]);
set(gca,'FontSize',11);

sgtitle('RQ3: Communication Performance of the Joint OFDM Waveform', ...
    'FontSize',15, 'FontWeight','bold');

outputName = 'RQ3_BER_vs_SNR_with_theory.png';
exportgraphics(fig, outputName, 'Resolution',300);
fprintf('Saved: %s\n', outputName);
