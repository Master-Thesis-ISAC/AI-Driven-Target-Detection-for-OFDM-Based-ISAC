function objects = generateObjects(numObjects, opts)
% generateObjects  Sample RCS-based targets for an ISAC scene.
%
% Targets are labelled by physical size category:
%   small  : RCS 0.1 - 5 m^2
%   medium : RCS 5 - 50 m^2
%   large  : RCS 50 - 500 m^2
%
% Usage:
%   objects = generateObjects(0);
%   objects = generateObjects(3, struct('sizeMix',[1 1 1], 'pMoving', 0.5));
%
% opts fields (all optional):
%   sizeMix     [pSmall pMedium pLarge] sampling weights (default [1 1 1])
%   xRange      [xmin xmax] m           (default [10 180])
%   yRange      [ymin ymax] m           (default [-100 100])
%   pMoving     P(target is moving)     (default 0.4)
%   vMaxSmall   max speed for small     (default 5 m/s)
%   vMaxMedium  max speed for medium    (default 25 m/s)
%   vMaxLarge   max speed for large     (default 0 m/s, static)

if nargin < 2 || isempty(opts); opts = struct(); end
xRange  = getf(opts, 'xRange',     [10 180]);
yRange  = getf(opts, 'yRange',     [-100 100]);
sizeMix = getf(opts, 'sizeMix',    [1 1 1]);
pMoving = getf(opts, 'pMoving',    0.4);
vMaxS   = getf(opts, 'vMaxSmall',  5);
vMaxM   = getf(opts, 'vMaxMedium', 25);
vMaxL   = getf(opts, 'vMaxLarge',  0);

sizeDef = struct( ...
    'name',    {'small',  'medium', 'large'}, ...
    'rcsLow',  {0.1,       5,        50    }, ...
    'rcsHigh', {5,         50,       500   }, ...
    'vMax',    {vMaxS,     vMaxM,    vMaxL }, ...
    'idx',     {1,         2,        3     });

emptyObj = struct('size',{}, 'sizeIdx',{}, 'position',{}, ...
                  'velocity',{}, 'rcs',{}, 'isMoving',{});
if numObjects <= 0
    objects = emptyObj;
    return;
end

objects = repmat(struct('size','', 'sizeIdx',0, ...
                        'position',[0;0;0], 'velocity',[0;0;0], ...
                        'rcs',0, 'isMoving',false), 1, numObjects);

w    = sizeMix(:) / sum(sizeMix);
cumW = cumsum(w);

for i = 1:numObjects
    r  = rand;
    s  = find(r <= cumW, 1, 'first');
    sd = sizeDef(s);

    objects(i).size    = sd.name;
    objects(i).sizeIdx = sd.idx;
    objects(i).position = [ ...
        xRange(1) + rand*(xRange(2) - xRange(1)); ...
        yRange(1) + rand*(yRange(2) - yRange(1)); ...
        0];

    % Log-uniform RCS sampling within the size band
    objects(i).rcs = sd.rcsLow * (sd.rcsHigh / sd.rcsLow)^rand;

    if sd.vMax > 0 && rand < pMoving
        speed   = rand * sd.vMax;
        heading = rand * 2*pi;
        objects(i).velocity = [cos(heading)*speed; sin(heading)*speed; 0];
        objects(i).isMoving = speed > 1e-3;
    else
        objects(i).velocity = [0;0;0];
        objects(i).isMoving = false;
    end
end
end


function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end