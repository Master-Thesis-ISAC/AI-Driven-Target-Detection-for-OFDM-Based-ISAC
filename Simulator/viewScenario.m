function viewScenario(filename)
% viewScenario  Load a saved scenario .mat file and display it.
%
% Usage:
%   viewScenario('output/scenario_0178.mat')
%   viewScenario       % opens a file picker

if nargin < 1 || isempty(filename)
    [f, p] = uigetfile('*.mat', 'Select a scenario file');
    if isequal(f, 0); return; end
    filename = fullfile(p, f);
end

S = load(filename);
N = double(S.numTargets);

hasFull3D = isfield(S, 'targetPosition_m') && isfield(S, 'targetVelocity_v');
hasSize   = isfield(S, 'targetSize');

objs = repmat(struct('size','','sizeIdx',0, ...
                     'position',[0;0;0],'velocity',[0;0;0], ...
                     'rcs',0,'isMoving',false), 1, N);

for i = 1:N
    if hasSize
        objs(i).size    = S.targetSize{i};
        objs(i).sizeIdx = double(S.targetSizeIdx(i));
    else
        rcs = S.targetRCS_m2(i);
        if     rcs <= 5;  objs(i).size = 'small';  objs(i).sizeIdx = 1;
        elseif rcs <= 50; objs(i).size = 'medium'; objs(i).sizeIdx = 2;
        else;             objs(i).size = 'large';  objs(i).sizeIdx = 3;
        end
    end
    objs(i).rcs      = S.targetRCS_m2(i);
    objs(i).isMoving = logical(S.targetIsMoving(i));
    if hasFull3D
        objs(i).position = S.targetPosition_m(:, i);
        objs(i).velocity = S.targetVelocity_v(:, i);
    else
        objs(i).position = [S.targetRange_m(i); 0; 0];
        objs(i).velocity = [-S.targetVel_mps(i); 0; 0];
    end
end

info                     = struct();
info.rangeAxis           = S.rangeAxis;
info.velocityAxis        = S.velocityAxis;
info.channelMeta.gNB_pos = [0;0;10];

m   = S.scenarioMeta;
ttl = sprintf('Scenario %d: %s (%d objects)  SNR_sense=%d dB', ...
              m.scenarioIndex, strrep(m.scenarioType,'_',' '), N, m.SNR_sense_dB);

visualizeScenario(objs, double(S.rdMap), info, ttl);

if ~hasFull3D
    warning('viewScenario:oldFormat', ...
        'File lacks targetPosition_m. Targets plotted along x-axis only.');
end
end
