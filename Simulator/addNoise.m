function rxNoisy = addNoise(rxClean, cfg, snr_dB)
% addNoise  Add complex AWGN to the sensing receive signal.
%
% Noise variance is set so the noise floor is consistent across all
% scenarios (present and absent). A unit reference power is used,
% matching the normalised transmit waveform in isacOFDMWaveform.
%
% For legacy callers that pass a real magnitude RD map the noise is
% added in the magnitude domain via legacyMagNoise.

if isreal(rxClean)
    rxNoisy = legacyMagNoise(rxClean, cfg, snr_dB);
    return;
end

referencePow = 1.0;

if nargin >= 3 && ~isempty(snr_dB)
    sigmaSq = referencePow / 10^(snr_dB / 10);
else
    sigmaSq = cfg.noisePower_lin;
    if isfield(cfg, 'SNR_sense_dB')
        sigmaSq = referencePow / 10^(cfg.SNR_sense_dB / 10);
    end
end
sigmaSq = max(sigmaSq, eps);

n       = sqrt(sigmaSq/2) * (randn(size(rxClean)) + 1j*randn(size(rxClean)));
rxNoisy = rxClean + n;

% Low-level urban clutter: sparse weak scatterers ~15 dB below reference
clutterDensity = 0.05;
nClutter       = round(clutterDensity * size(rxClean, 1));
if nClutter > 0
    clutterPow = referencePow * 10^(-15/10);
    cIdx = randperm(size(rxClean, 1), nClutter);
    cAmp = sqrt(clutterPow/2) * ...
           (randn(nClutter, size(rxClean,2)) + 1j*randn(nClutter, size(rxClean,2)));
    rxNoisy(cIdx, :) = rxNoisy(cIdx, :) + cAmp;
end
end


function noisyMap = legacyMagNoise(rdMap, cfg, snr_dB) %#ok<INUSD>
% Noise addition for real magnitude RD maps (legacy callers only).
mapPow = mean(rdMap(:).^2);
if mapPow <= 0
    noisyMap = rdMap;
    return;
end
target_dB = 20;
if exist('snr_dB', 'var') && ~isempty(snr_dB); target_dB = snr_dB; end
sigma    = sqrt(mapPow / 10^(target_dB/10) / 2);
noisyMap = abs(rdMap + sigma * (randn(size(rdMap)) + 1j*randn(size(rdMap))));
end