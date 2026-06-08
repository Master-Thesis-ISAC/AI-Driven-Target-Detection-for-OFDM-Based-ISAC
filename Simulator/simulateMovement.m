function objects = simulateMovement(objects, dt, sceneBounds)
% simulateMovement  Advance target positions by dt seconds at constant velocity.
% dt should equal cfg.T_CPI_s so saved labels match the received signal timing.
%
% Usage:
%   objects = simulateMovement(objects, cfg.T_CPI_s);
%   objects = simulateMovement(objects, cfg.T_CPI_s, struct('x',[10 180],'y',[-100 100]));

if nargin < 2 || isempty(dt);          dt = 4e-3;          end
if nargin < 3 || isempty(sceneBounds); sceneBounds = struct(); end
xLim = getf(sceneBounds, 'x', [10 180]);
yLim = getf(sceneBounds, 'y', [-100 100]);

if isempty(objects); return; end

for i = 1:numel(objects)
    if ~objects(i).isMoving; continue; end

    objects(i).position = objects(i).position + objects(i).velocity * dt;

    % Reflect velocity at scene boundary to keep targets inside the RD map extent
    p = objects(i).position;
    v = objects(i).velocity;
    if p(1) < xLim(1) || p(1) > xLim(2)
        v(1) = -v(1);
        p(1) = max(xLim(1), min(xLim(2), p(1)));
    end
    if p(2) < yLim(1) || p(2) > yLim(2)
        v(2) = -v(2);
        p(2) = max(yLim(1), min(yLim(2), p(2)));
    end
    objects(i).position = p;
    objects(i).velocity = v;
end
end


function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)); v = s.(f); else; v = d; end
end