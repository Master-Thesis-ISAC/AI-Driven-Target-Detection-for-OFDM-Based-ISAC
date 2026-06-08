function saveData(rdMap, objects, info, meta, fileName)
% saveData  Save one ISAC scenario to a .mat file for AI training.
%
% Saves rdMap, ground-truth labels, comm metrics, classical detector
% outputs, and axis/parameter structs. Scene-level size label is the
% largest target size present in the scene.

rdMap    = single(rdMap);
rdMap_dB = single(20*log10(double(rdMap) + eps));

N = numel(objects);
if N == 0
    targetSize       = {};
    targetSizeIdx    = zeros(1,0,'int32');
    targetRange_m    = zeros(1,0);
    targetVel_mps    = zeros(1,0);
    targetRCS_m2     = zeros(1,0);
    targetIsMoving   = false(1,0);
    targetPosition_m = zeros(3,0);
    targetVelocity_v = zeros(3,0);
    sceneLargestSize = int32(0);
else
    if isfield(info,'channelMeta') && isfield(info.channelMeta,'gNB_pos')
        gNB = info.channelMeta.gNB_pos;
    elseif isfield(info,'gNB_pos')
        gNB = info.gNB_pos;
    else
        gNB = [0;0;10];
    end
    pos              = reshape([objects.position], 3, N);
    vel              = reshape([objects.velocity], 3, N);
    rel              = pos - gNB;
    R                = sqrt(sum(rel.^2, 1));
    losU             = rel ./ max(R, 1e-6);
    targetRange_m    = R;
    targetVel_mps    = sum(vel .* losU, 1);
    targetRCS_m2     = [objects.rcs];
    targetIsMoving   = [objects.isMoving];
    targetSize       = {objects.size};
    targetSizeIdx    = int32([objects.sizeIdx]);
    targetPosition_m = pos;
    targetVelocity_v = vel;
    sceneLargestSize = int32(max(targetSizeIdx));
end
numTargets   = N;
targetPresent = N > 0;

rangeAxis    = info.rangeAxis;
velocityAxis = info.velocityAxis;

radarParams = struct( ...
    'fc_Hz',      info.fc_Hz,      'BW_Hz',      info.BW_Hz, ...
    'lambda_m',   info.lambda_m,   'rangeRes_m', info.rangeRes_m, ...
    'rangeBin_m', info.rangeBin_m, 'velRes_mps', info.velRes_mps, ...
    'velBin_mps', info.velBin_mps, 'maxRange_m', info.maxRange_m, ...
    'maxVel_mps', info.maxVel_mps);

channelParams = struct( ...
    'processingMode', getf(info,'processingMode','unknown'), ...
    'Nr_ant',         getf(info,'Nr_ant', 1), ...
    'Nsc',            getf(info,'Nsc',    0), ...
    'Nsym',           getf(info,'Nsym',   0));
if isfield(info,'channelMeta')
    channelParams.channelModel = info.channelMeta.channelModel;
    channelParams.Nt           = info.channelMeta.Nt;
    channelParams.Nr_comm      = info.channelMeta.Nr_comm;
end

precodingParams = struct('beamformGain',1,'condNumber',1,'numLayers',1);
if isfield(info,'precodingInfo')
    p = info.precodingInfo;
    precodingParams.beamformGain = getf(p,'beamformGain',1);
    precodingParams.condNumber   = getf(p,'condNumber',  1);
    precodingParams.numLayers    = getf(p,'numLayers',   1);
    if isfield(p,'singularVals')
        precodingParams.singularVals = p.singularVals(:).';
    end
end

scenarioMeta = meta;

if isfield(info,'commMetrics') && ~isempty(info.commMetrics)
    BER          = info.commMetrics.BER;
    EVM_dB       = info.commMetrics.EVM_dB;
    numBitsTx    = info.commMetrics.numBitsTx;
    numBitErrors = info.commMetrics.numBitErrors;
    SNR_eff_dB   = info.commMetrics.SNR_eff_dB;
else
    BER=NaN; EVM_dB=NaN; numBitsTx=0; numBitErrors=0; SNR_eff_dB=NaN;
end

if isfield(info,'classical') && ~isempty(info.classical)
    classicalCFAR_present     = info.classical.CFAR_present;
    classicalCFAR_numDetected = info.classical.CFAR_numDetected;
    classicalCFAR_peaks       = info.classical.CFAR_peaks;
    classicalEnergy_present   = info.classical.Energy_present;
    classicalEnergy_dB        = info.classical.Energy_dB;
    classicalCFAR_Pfa_grid    = info.classical.CFAR_Pfa_grid;
    classicalCFAR_grid        = info.classical.CFAR_present_grid;
else
    classicalCFAR_present=false; classicalCFAR_numDetected=0;
    classicalCFAR_peaks=zeros(0,2); classicalEnergy_present=false;
    classicalEnergy_dB=NaN; classicalCFAR_Pfa_grid=[]; classicalCFAR_grid=[];
end

if isfield(info,'targetRangeBin') && ~isempty(info.targetRangeBin)
    targetRangeBin = int32(info.targetRangeBin);
    targetDoppBin  = int32(info.targetDoppBin);
else
    targetRangeBin = zeros(1,0,'int32');
    targetDoppBin  = zeros(1,0,'int32');
end

[outDir, ~, ~] = fileparts(fileName);
if ~isempty(outDir) && ~exist(outDir,'dir'); mkdir(outDir); end

save(fileName, ...
    'rdMap', 'rdMap_dB', ...
    'targetPresent', 'sceneLargestSize', ...
    'targetSize', 'targetSizeIdx', ...
    'targetRange_m', 'targetVel_mps', 'targetRCS_m2', 'targetIsMoving', ...
    'targetPosition_m', 'targetVelocity_v', ...
    'targetRangeBin', 'targetDoppBin', ...
    'numTargets', 'rangeAxis', 'velocityAxis', ...
    'BER', 'EVM_dB', 'numBitsTx', 'numBitErrors', 'SNR_eff_dB', ...
    'classicalCFAR_present', 'classicalCFAR_numDetected', ...
    'classicalCFAR_peaks', 'classicalEnergy_present', 'classicalEnergy_dB', ...
    'classicalCFAR_Pfa_grid', 'classicalCFAR_grid', ...
    'radarParams', 'channelParams', 'precodingParams', 'scenarioMeta', ...
    '-v7.3');

fprintf('[saveData] %04d -> %s  (%d targets, %s)\n', ...
    meta.scenarioIndex, fileName, N, channelParams.processingMode);
end


function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end
