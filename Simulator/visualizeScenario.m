function visualizeScenario(objects, rdMap, info, scenarioTitle)
% visualizeScenario  Three-panel diagnostic plot for one ISAC scenario.
%
% Panel 1: Scene layout (cross-range vs range)
% Panel 2: RD map in dB with ground-truth target positions overlaid
% Panel 3: Per-target range and radial velocity stem plot

if nargin < 4; scenarioTitle = 'ISAC Scenario'; end

persistent markerMap markerSize
if isempty(markerMap)
    markerMap  = containers.Map({'small','medium','large'}, {'g.','bs','rp'});
    markerSize = containers.Map({'small','medium','large'}, {8, 10, 14});
end

fig = figure('Name', scenarioTitle, 'NumberTitle','off', ...
             'Color','w', 'Position',[100 100 1500 460]);

% Panel 1: scene layout
ax1 = subplot(1,3,1,'Parent',fig);
hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');

plot(ax1, 0, 0, 'ko', 'MarkerFaceColor','k', 'MarkerSize',10, 'DisplayName','gNB');

if isfield(info,'channelMeta') && isfield(info.channelMeta,'UE_pos')
    ue = info.channelMeta.UE_pos;
else
    ue = [100; 30; 1.5];
end
plot(ax1, ue(2), ue(1), 'b^', 'MarkerFaceColor','b', 'MarkerSize',10, 'DisplayName','UE');
plot(ax1, [0 ue(2)], [0 ue(1)], 'b:', 'LineWidth',1, 'HandleVisibility','off');
text(ax1, ue(2)+2, ue(1)+2, 'UE', 'FontSize',9, 'Color','b');

placed = containers.Map();
for i = 1:numel(objects)
    o       = objects(i);
    cr      = o.position(2);
    r       = o.position(1);
    sizeKey = 'medium';
    if isfield(o,'size') && ~isempty(o.size); sizeKey = o.size; end
    mk  = ternary(isKey(markerMap, sizeKey),  markerMap(sizeKey),  'cs');
    msz = ternary(isKey(markerSize,sizeKey),  markerSize(sizeKey), 10);
    if ~isKey(placed, sizeKey)
        plot(ax1, cr, r, mk, 'MarkerSize',msz, 'LineWidth',1.5, 'DisplayName',sizeKey);
        placed(sizeKey) = true;
    else
        plot(ax1, cr, r, mk, 'MarkerSize',msz, 'LineWidth',1.5, 'HandleVisibility','off');
    end
    if o.isMoving && norm(o.velocity) > 0.1
        quiver(ax1, cr, r, o.velocity(2)*3, o.velocity(1)*3, 0, ...
               'Color',[0.4 0.4 0.4], 'MaxHeadSize',0.5, 'HandleVisibility','off');
    end
    text(ax1, cr+2, r+2, sprintf('%s %d',sizeKey,i), 'FontSize',8, 'Color',[0.3 0.3 0.3]);
end
xlabel(ax1,'Cross-range y (m)'); ylabel(ax1,'Range x (m)');
title(ax1,'Scenario Layout');
xlim(ax1,[-120 120]); ylim(ax1,[-20 220]);
legend(ax1,'Location','northeast');

% Panel 2: RD map
ax2          = subplot(1,3,2,'Parent',fig);
rangeAxis    = info.rangeAxis;
velocityAxis = info.velocityAxis;
rdLog        = 20*log10(rdMap + eps);
peakDb       = max(rdLog(:));
rdLog        = max(rdLog, peakDb - 50);   % clip to top 50 dB for display

imagesc(ax2, velocityAxis, rangeAxis, rdLog.');
axis(ax2,'xy'); colormap(ax2,'jet');
cb = colorbar(ax2); ylabel(cb,'Magnitude (dB)');
xlabel(ax2,'Radial velocity (m/s)'); ylabel(ax2,'Range (m)');
title(ax2,'Range-Doppler Map');

if ~isempty(objects)
    gNB = [0;0;10];
    if isfield(info,'channelMeta'); gNB = info.channelMeta.gNB_pos; end
    hold(ax2,'on');
    for i = 1:numel(objects)
        o   = objects(i);
        rel = o.position - gNB;
        R   = norm(rel);
        vr  = ternary(R>0, dot(o.velocity, rel/R), 0);
        plot(ax2, vr, R, 'wx', 'MarkerSize',10, 'LineWidth',2, 'HandleVisibility','off');
        lbl = ''; if isfield(o,'size'); lbl = o.size; end
        text(ax2, vr+0.3, R+2, lbl, 'Color','w', 'FontSize',7);
    end
end

% Panel 3: range/velocity stems
ax3 = subplot(1,3,3,'Parent',fig);
N   = numel(objects);
if N == 0
    text(ax3, 0.5, 0.5, 'No targets', 'Units','normalized', ...
         'HorizontalAlignment','center', 'FontSize',12, 'Color',[0.5 0.5 0.5]);
    title(ax3,'Target Range & Velocity'); axis(ax3,'off'); return;
end
gNB = [0;0;10];
if isfield(info,'channelMeta'); gNB = info.channelMeta.gNB_pos; end
R = zeros(1,N); vr = zeros(1,N); lbl = cell(1,N);
for i = 1:N
    rel   = objects(i).position - gNB;
    R(i)  = norm(rel);
    vr(i) = ternary(R(i)>0, dot(objects(i).velocity, rel/R(i)), 0);
    lbl{i} = ''; if isfield(objects(i),'size'); lbl{i} = objects(i).size; end
end

yyaxis(ax3,'left');
stem(ax3, 1:N, R, 'filled', 'LineWidth',1.5, 'MarkerSize',6, ...
     'Color',[0.18 0.50 0.75], 'MarkerFaceColor',[0.18 0.50 0.75]);
ylabel(ax3,'Range (m)','Color',[0.18 0.50 0.75]);
ax3.YColor = [0.18 0.50 0.75];
ylim(ax3,[0, max(R)*1.3+1]);

yyaxis(ax3,'right');
stem(ax3, 1:N, vr, 'filled', 'LineWidth',1.5, 'MarkerSize',6, ...
     'Color',[0.85 0.33 0.10], 'MarkerFaceColor',[0.85 0.33 0.10]);
ylabel(ax3,'Radial velocity (m/s)','Color',[0.85 0.33 0.10]);
ax3.YColor = [0.85 0.33 0.10];

yyaxis(ax3,'left');
for i = 1:N
    text(ax3, i, R(i)+max(R)*0.04+0.5, lbl{i}, 'FontSize',8, 'HorizontalAlignment','center');
end
xticks(ax3,1:N); xticklabels(ax3, compose('T%d',1:N));
xlabel(ax3,'Target index'); title(ax3,'Target Range & Velocity');
grid(ax3,'on'); xlim(ax3,[0.5 N+0.5]);

sgtitle(fig, scenarioTitle, 'FontWeight','bold');
drawnow;
end


function v = ternary(c, a, b); if c; v = a; else; v = b; end; end
