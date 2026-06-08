function [precodedWaveform, precodingInfo] = applySVDPrecoding(txWaveform, H, numLayers)
% applySVDPrecoding  SVD-based transmit beamforming.
%
% For channel estimate H = U*S*V^H, the precoder is F = V(:,1:numLayers).
% The precoded waveform is X[n,:] = s[n] * F.' with total Tx power = numLayers.
%
% Inputs:
%   txWaveform : [Nsamples x 1] single-stream waveform
%   H          : [Nr x Nt] channel estimate (use eye(Nt) for no-CSI operation)
%   numLayers  : number of spatial layers (default 1)

if nargin < 3 || isempty(numLayers); numLayers = 1; end

if isempty(H) || ~all(isfinite(H), 'all')
    % No CSI: equal-power broadcast across all antennas
    Nt = 4;
    F  = ones(Nt, 1) / sqrt(Nt);
    precodedWaveform = txWaveform(:,1) * F.';
    precodingInfo = struct('F',F, 'singularVals',1, 'condNumber',1, ...
                           'numLayers',1, 'beamformGain',1, 'Nt',Nt, 'Nr',Nt);
    return;
end

[Nr, Nt]  = size(H);
numLayers = max(1, min([numLayers, Nr, Nt]));

[~, S, V] = svd(H, 'econ');
sv = diag(S);
F  = V(:, 1:numLayers);

sIn              = repmat(txWaveform, 1, numLayers);
precodedWaveform = sIn * F.';

% Normalise total Tx power to numLayers
totPow = mean(sum(abs(precodedWaveform).^2, 2));
if totPow > 0
    precodedWaveform = precodedWaveform * sqrt(numLayers / totPow);
end

condNumber   = sv(1) / max(sv(end), 1e-12);
beamformGain = sum(sv(1:numLayers).^2);

precodingInfo = struct( ...
    'F',            F, ...
    'singularVals', sv, ...
    'condNumber',   condNumber, ...
    'numLayers',    numLayers, ...
    'beamformGain', beamformGain, ...
    'Nt',           Nt, ...
    'Nr',           Nr);
end