% GenerateISACData_Main.m
% Generates the 5000-scenario OFDM-ISAC dataset used in the thesis.
% Each scenario runs the full pipeline: waveform -> channel -> RD map -> save.
% Outputs scenario_XXXX.mat files and manifest.csv in the output folder.

clear; clc; close all;

% --- Configuration ---
outputDir        = 'output';
numScenariosEach = 500;       % per template; 6 templates x 500 = 3000 present
doVisualize      = false;
doSave           = true;

splitFrac.train = 0.70;
splitFrac.val   = 0.15;
splitFrac.test  = 0.15;

UE_x_range = [80  140];   % m
UE_y_range = [-40  40];   % m
UE_z_fixed = 1.5;         % m

SNR_sense_grid_dB = [0, 5, 10, 15, 20, 25, 30];
SNR_comm_grid_dB  = [5, 10, 15, 20, 25];

baseCfg = buildWaveformConfig( ...
    'fc_GHz',         3.5, ...
    'BW_MHz',         100, ...
    'SCS_kHz',        30,  ...
    'modOrder',       16,  ...
    'numSlots',       16,  ...
    'Nt',             4,   ...
    'Nr_comm',        2,   ...
    'Nr_sense',       4,   ...
    'numLayers',      1,   ...
    'numRangeBins',   128, ...
    'numDopplerBins', 64);

dt_CPI = baseCfg.T_CPI_s;

harq = HARQEntity('PacketErrorRate', 0.10, 'MaxRetx', 3);

% --- Scenario templates ---
% { name, minObj, maxObj, sizeMix [pSmall pMedium pLarge], pMoving }
templates = { ...
    'empty',         0, 0, [0 0 0], 0.0 ; ...
    'single_small',  1, 1, [1 0 0], 0.6 ; ...
    'single_medium', 1, 1, [0 1 0], 0.5 ; ...
    'single_large',  1, 1, [0 0 1], 0.0 ; ...
    'multi_small',   2, 4, [1 0 0], 0.6 ; ...
    'multi_mixed',   2, 5, [1 1 1], 0.4 };

% Empty template runs 5x to keep the dataset 50/50 present/absent.
emptyMultiplier = 5;

if ~exist(outputDir, 'dir'); mkdir(outputDir); end

% --- Generation loop ---
manifest    = {};
scenarioIdx = 0;

for tIdx = 1:size(templates, 1)
    sType   = templates{tIdx, 1};
    minObj  = templates{tIdx, 2};
    maxObj  = templates{tIdx, 3};
    sizeMix = templates{tIdx, 4};
    pMoving = templates{tIdx, 5};

    if strcmp(sType, 'empty')
        repsThisType = numScenariosEach * emptyMultiplier;
    else
        repsThisType = numScenariosEach;
    end

    for rep = 1:repsThisType
        scenarioIdx = scenarioIdx + 1;
        seed        = scenarioIdx;
        rng(seed);

        % Cycle through SNR grids to ensure balanced coverage
        snr_s_idx = mod(rep-1, numel(SNR_sense_grid_dB)) + 1;
        snr_c_idx = mod(rep-1, numel(SNR_comm_grid_dB))  + 1;

        cfg              = baseCfg;
        cfg.SNR_sense_dB = SNR_sense_grid_dB(snr_s_idx);
        cfg.SNR_comm_dB  = SNR_comm_grid_dB(snr_c_idx);

        ue_x     = UE_x_range(1) + rand * (UE_x_range(2) - UE_x_range(1));
        ue_y     = UE_y_range(1) + rand * (UE_y_range(2) - UE_y_range(1));
        cfg.UE_pos = [ue_x; ue_y; UE_z_fixed];

        fprintf('Scenario %04d  [%s %d/%d  SNR_s=%ddB  SNR_c=%ddB]\n', ...
            scenarioIdx, sType, rep, repsThisType, ...
            cfg.SNR_sense_dB, cfg.SNR_comm_dB);

        % Generate targets
        if minObj == 0 && maxObj == 0
            objects = generateObjects(0);
        else
            n       = randi([minObj, maxObj]);
            objects = generateObjects(n, struct('sizeMix', sizeMix, 'pMoving', pMoving));
        end
        objects = simulateMovement(objects, dt_CPI);

        % Run ISAC pipeline
        [rd, info]   = simulateISAC(objects, cfg, struct('seed', seed));
        rdCombined   = harq.chaseCombine(rd);

        meta = struct( ...
            'scenarioIndex', scenarioIdx, ...
            'scenarioType',  sType, ...
            'numObjects',    numel(objects), ...
            'repIndex',      rep, ...
            'seed',          seed, ...
            'SNR_sense_dB',  cfg.SNR_sense_dB, ...
            'SNR_comm_dB',   cfg.SNR_comm_dB, ...
            'UE_x',          ue_x, ...
            'UE_y',          ue_y, ...
            'waveformMode',  info.processingMode);

        BER_val   = getfStruct(info, 'commMetrics', 'BER',            NaN);
        EVM_val   = getfStruct(info, 'commMetrics', 'EVM_dB',         NaN);
        cfar_pres = getfStruct(info, 'classical',   'CFAR_present',   false);
        ene_pres  = getfStruct(info, 'classical',   'Energy_present', false);

        if doSave
            fName = fullfile(outputDir, sprintf('scenario_%04d.mat', scenarioIdx));
            saveData(rdCombined, objects, info, meta, fName);

            r = rand;
            if     r < splitFrac.train
                split = 'train';
            elseif r < splitFrac.train + splitFrac.val
                split = 'val';
            else
                split = 'test';
            end

            if numel(objects) > 0
                largestSize = max([objects.sizeIdx]);
            else
                largestSize = 0;
            end

            manifest(end+1, :) = { ...
                scenarioIdx, sType, numel(objects), ...
                double(numel(objects) > 0), ...
                double(largestSize), ...
                cfg.SNR_sense_dB, cfg.SNR_comm_dB, ...
                BER_val, EVM_val, ...
                double(cfar_pres), double(ene_pres), ...
                ue_x, ue_y, ...
                seed, split, fName }; %#ok<SAGROW>
        end

        if doVisualize
            ttl = sprintf('Scenario %d: %s (%d objects)', ...
                scenarioIdx, strrep(sType, '_', ' '), numel(objects));
            visualizeScenario(objects, rdCombined, info, ttl);
        end
    end
end

% --- Write manifest ---
if doSave && ~isempty(manifest)
    manifestT = cell2table(manifest, 'VariableNames', ...
        {'idx', 'type', 'numObj', 'targetPresent', 'sceneLargestSize', ...
         'SNR_sense_dB', 'SNR_comm_dB', 'BER', 'EVM_dB', ...
         'CFAR_present', 'Energy_present', ...
         'UE_x', 'UE_y', 'seed', 'split', 'file'});
    writetable(manifestT, fullfile(outputDir, 'manifest.csv'));
end

fprintf('\nDone. %d scenarios saved to "%s/"\n', scenarioIdx, outputDir);


function v = getfStruct(s, f1, f2, d)
% Returns s.f1.f2 if it exists, otherwise returns default d.
if isfield(s, f1) && isstruct(s.(f1)) && isfield(s.(f1), f2)
    v = s.(f1).(f2);
    if isempty(v); v = d; end
else
    v = d;
end
end
