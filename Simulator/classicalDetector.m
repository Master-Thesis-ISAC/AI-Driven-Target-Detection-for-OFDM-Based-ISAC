function out = classicalDetector(rdMap, info, opts)
% classicalDetector  CA-CFAR and energy detector on an RD map.
%
% CA-CFAR is evaluated at five Pfa settings for ROC analysis.
% The median-Pfa decision is returned as the main CFAR_present output.
% A connected-cluster requirement (minClusterSize = 5) rejects noise spikes.
%
% Usage:
%   out = classicalDetector(rdMap, info)
%   out = classicalDetector(rdMap, info, struct('Pfa_grid', [1e-6 1e-8]))

if nargin < 3 || isempty(opts); opts = struct(); end

Pfa_grid        = getf(opts, 'Pfa_grid',        [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]);
guardCells      = getf(opts, 'guardCells',      [2 2]);
trainCells      = getf(opts, 'trainCells',      [4 8]);
energyThresh_dB = getf(opts, 'energyThresh_dB', 6);
minClusterSize  = getf(opts, 'minClusterSize',  5);

[Nd, Nr] = size(rdMap);
mapPow   = rdMap.^2;
noiseFloor = median(mapPow(:));

% CA-CFAR
gD = guardCells(1);  gR = guardCells(2);
tD = trainCells(1);  tR = trainCells(2);

winFull   = (2*(gD+tD)+1) * (2*(gR+tR)+1);
winGuard  = (2*gD+1)      * (2*gR+1);
N_train   = winFull - winGuard;

fullKernel  = ones(2*(gD+tD)+1, 2*(gR+tR)+1);
guardKernel = padarray(ones(2*gD+1, 2*gR+1), [tD tR]);
trainKernel = (fullKernel - guardKernel) / N_train;
noiseEst    = conv2(mapPow, trainKernel, 'same');

nPfa              = numel(Pfa_grid);
CFAR_present_grid = false(1, nPfa);
CFAR_count_grid   = zeros(1, nPfa);
alpha_grid        = zeros(1, nPfa);
rBin_main         = [];
dBin_main         = [];

for p = 1:nPfa
    alpha_p   = N_train * (Pfa_grid(p)^(-1/N_train) - 1);
    cfarHits  = mapPow > alpha_p * noiseEst;

    % Suppress map edges where the CFAR window overhangs
    em_d = gD + tD;  em_r = gR + tR;
    cfarHits(1:em_d, :)           = false;
    cfarHits(end-em_d+1:end, :)   = false;
    cfarHits(:, 1:em_r)           = false;
    cfarHits(:, end-em_r+1:end)   = false;

    [dBin, rBin] = find(cfarHits);

    if ~isempty(dBin)
        try
            cc           = bwconncomp(cfarHits, 8);
            clusterSizes = cellfun(@numel, cc.PixelIdxList);
        catch
            clusterSizes = manualConnComp(cfarHits);
        end
        nValid = sum(clusterSizes >= minClusterSize);
    else
        nValid = 0;
    end

    CFAR_present_grid(p) = nValid > 0;
    CFAR_count_grid(p)   = nValid;
    alpha_grid(p)        = alpha_p;

    if p == ceil(nPfa/2)
        rBin_main = rBin;
        dBin_main = dBin;
    end
end

% Energy detector
totalEnergy = sum(mapPow(:));
floorEnergy = noiseFloor * Nd * Nr;
energy_dB   = 10*log10(totalEnergy / max(floorEnergy, eps));

midIdx = ceil(nPfa/2);
out = struct( ...
    'CFAR_present',          CFAR_present_grid(midIdx), ...
    'CFAR_numDetected',      CFAR_count_grid(midIdx), ...
    'CFAR_peaks',            [rBin_main dBin_main], ...
    'CFAR_threshold_alpha',  alpha_grid(midIdx), ...
    'CFAR_Pfa',              Pfa_grid(midIdx), ...
    'CFAR_Pfa_grid',         Pfa_grid, ...
    'CFAR_present_grid',     CFAR_present_grid, ...
    'CFAR_count_grid',       CFAR_count_grid, ...
    'CFAR_alpha_grid',       alpha_grid, ...
    'Energy_present',        energy_dB > energyThresh_dB, ...
    'Energy_dB',             energy_dB, ...
    'Energy_thresh_dB',      energyThresh_dB, ...
    'noiseFloor',            noiseFloor);
end


function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end


function sizes = manualConnComp(mask)
% 8-connected component sizes. Fallback when Image Processing Toolbox is absent.
visited = false(size(mask));
sizes   = [];
[Nr, Nc] = size(mask);
for r = 1:Nr
    for c = 1:Nc
        if mask(r,c) && ~visited(r,c)
            sz    = 0;
            stack = [r c];
            while ~isempty(stack)
                rr = stack(end,1); cc = stack(end,2);
                stack(end,:) = [];
                if rr<1||rr>Nr||cc<1||cc>Nc; continue; end
                if visited(rr,cc)||~mask(rr,cc); continue; end
                visited(rr,cc) = true;
                sz = sz + 1;
                for dr = -1:1
                    for dc = -1:1
                        if dr~=0||dc~=0
                            stack(end+1,:) = [rr+dr cc+dc]; %#ok<AGROW>
                        end
                    end
                end
            end
            sizes(end+1) = sz; %#ok<AGROW>
        end
    end
end
end
