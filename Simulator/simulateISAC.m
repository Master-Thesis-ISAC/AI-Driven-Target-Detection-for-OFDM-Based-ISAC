function [rdMap, info] = simulateISAC(objects, cfg, opts)
% simulateISAC  Run one ISAC frame through the full pipeline.
%
% Pipeline stages:
%   1. OFDM waveform generation
%   2. SVD precoding (probe pass to estimate H, then precoded Tx)
%   3. TR 38.901 channel (comm path + target echoes)
%   4. AWGN at receive front-end
%   5. Sensing processor -> RD map
%   6. UE communication receiver -> BER/EVM
%   7. Classical CA-CFAR detector
%
% Usage:
%   [rd, info] = simulateISAC(objects, cfg, struct('seed', 42));

if nargin < 3 || isempty(opts); opts = struct(); end

if isfield(opts, 'seed') && ~isempty(opts.seed)
    rngStream = RandStream('mt19937ar', 'Seed', opts.seed);
else
    rngStream = RandStream('mt19937ar', 'Seed', 'shuffle');
end

useFallback   = isfield(opts, 'useFallback')   && opts.useFallback;
allowFallback = isfield(opts, 'allowFallback') && opts.allowFallback;

if useFallback
    [rdMap, info] = pointTargetFallback(objects, cfg, rngStream);
    info.processingMode = 'fallback_pointtarget';
    info = attachGroundTruth(info, objects, cfg);
    return;
end

try
    % Stage 1: OFDM waveform
    [s_tx, waveformInfo] = isacOFDMWaveform(cfg, rngStream);

    % Stage 2: Probe pass to estimate comm channel H, then SVD precoding
    s_probe = repmat(s_tx, 1, cfg.Nt) / sqrt(cfg.Nt);
    [~, H_comm_est, ~] = h38901ISACChannel(s_probe, waveformInfo, objects, cfg, 'comm');
    [X_tx, precodingInfo] = applySVDPrecoding(s_tx, H_comm_est, cfg.numLayers);

    % Stage 3: Full channel (comm + sensing echoes)
    [rxSensing, H_comm, channelMeta] = h38901ISACChannel(X_tx, waveformInfo, objects, cfg, 'full');

    % Stage 4: Receiver noise
    rxNoisy = addNoise(rxSensing, cfg);

    % Stage 5: RD map
    [rdMap, sensingInfo] = isacSensingProcessor(rxNoisy, X_tx, waveformInfo, cfg);

    % Stage 6: UE communication receiver
    commMetrics = ueDecoder(X_tx, waveformInfo, H_comm, cfg);

    % Stage 7: Classical detector
    classical = classicalDetector(rdMap, sensingInfo);

    info                = sensingInfo;
    info.channelMatrix  = H_comm;
    info.precodingInfo  = precodingInfo;
    info.channelMeta    = channelMeta;
    info.commMetrics    = commMetrics;
    info.classical      = classical;
    info.processingMode = 'full_OFDM_TR38901';

catch ME
    fprintf(2, '\n[simulateISAC] ERROR: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  at %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
    if ~allowFallback
        rethrow(ME);
    end
    warning('simulateISAC:fallback', ...
        'Full chain failed. Using point-target fallback (allowFallback=true).');
    [rdMap, info] = pointTargetFallback(objects, cfg, rngStream);
    info.processingMode = 'fallback_pointtarget';
end

info = attachGroundTruth(info, objects, cfg);
end


function [rd, ci] = pointTargetFallback(objects, cfg, rngStream)
% Minimal point-target RD map. Used only if the full chain throws.
c        = 299792458;
Nr       = cfg.numRangeBins;
Nd       = cfg.numDopplerBins;
lambda   = cfg.lambda_m;
rangeRes = c / (2 * cfg.BW_Hz);
maxRange = rangeRes * Nr;
maxVel   = lambda * cfg.SCS_Hz / 4;

raw = complex(zeros(Nd, Nr));
for i = 1:numel(objects)
    o = objects(i);
    R = norm(o.position - cfg.gNB_pos);
    if R < 1 || R > maxRange; continue; end
    losU = (o.position - cfg.gNB_pos) / R;
    vr   = dot(o.velocity, losU);
    if abs(vr) > maxVel; continue; end
    rB  = R / rangeRes;
    dB  = (vr / maxVel) * (Nd / 2);
    amp = sqrt(o.rcs / R^4);
    k   = (0:Nr-1);
    m   = (0:Nd-1).';
    raw = raw + amp * exp(1j*2*pi*(dB/Nd*m + rB/Nr*k));
end
n   = (randn(rngStream, Nd, Nr) + 1j*randn(rngStream, Nd, Nr)) / sqrt(2);
raw = raw + n * 1e-3;
rd  = abs(fftshift(fft2(raw), 1));

ci = struct( ...
    'rangeAxis',    (0:Nr-1)*rangeRes, ...
    'velocityAxis', linspace(-maxVel, maxVel, Nd), ...
    'rangeRes_m',   rangeRes, ...
    'rangeBin_m',   rangeRes, ...
    'velRes_mps',   2*maxVel/Nd, ...
    'velBin_mps',   2*maxVel/Nd, ...
    'maxRange_m',   maxRange, ...
    'maxVel_mps',   maxVel, ...
    'fc_Hz',        cfg.fc_Hz, ...
    'BW_Hz',        cfg.BW_Hz, ...
    'lambda_m',     lambda, ...
    'Nsc',          Nr, ...
    'Nsym',         Nd, ...
    'Nr_ant',       1);
end


function info = attachGroundTruth(info, objects, cfg)
% Attach ground-truth range, velocity, size and bin indices to info struct.
gNB = cfg.gNB_pos;
N   = numel(objects);
info.numTargets = N;

if N == 0
    info.targetRanges_m = [];
    info.targetVels_mps = [];
    info.targetSize     = {};
    info.targetSizeIdx  = [];
    info.targetRCS      = [];
    info.targetRangeBin = [];
    info.targetDoppBin  = [];
    return;
end

pos = reshape([objects.position], 3, N);
vel = reshape([objects.velocity], 3, N);
rel = pos - gNB;
R   = sqrt(sum(rel.^2, 1));
los = rel ./ max(R, 1e-6);
vr  = sum(vel .* los, 1);

info.targetRanges_m = R;
info.targetVels_mps = vr;
info.targetSize     = {objects.size};
info.targetSizeIdx  = [objects.sizeIdx];
info.targetRCS      = [objects.rcs];

if isfield(info, 'rangeAxis') && ~isempty(info.rangeAxis)
    rngAxis        = info.rangeAxis;
    velAxis        = info.velocityAxis;
    targetRangeBin = zeros(1, N);
    targetDoppBin  = zeros(1, N);
    for k = 1:N
        [~, targetRangeBin(k)] = min(abs(rngAxis - R(k)));
        [~, targetDoppBin(k)]  = min(abs(velAxis - vr(k)));
    end
    info.targetRangeBin = targetRangeBin;
    info.targetDoppBin  = targetDoppBin;
else
    info.targetRangeBin = [];
    info.targetDoppBin  = [];
end
end
