function tests = test_isac_pipeline
% test_isac_pipeline  Unit tests for the rewritten ISAC chain.
%
%   Run from the parent directory:
%       results = runtests('tests/test_isac_pipeline')
%
%   Tests
%   -----
%     test_config_values     : numRB, range/Doppler limits, kTBF noise
%     test_ofdm_roundtrip    : demod(mod(grid)) == grid (unit test, A3)
%     test_single_target_RD  : synthetic target → peak within one bin of GT
%     test_empty_scene       : numTargets==0 path produces a valid map
%     test_size_indices      : labels are integer 1..3 with matching strings
%     test_dataset_amplitude : RD map amplitudes preserve relative RCS

tests = functiontests(localfunctions);
end


function setupOnce(testCase)
% Add parent directory to path so module under test is resolvable.
here    = fileparts(mfilename('fullpath'));
parent  = fileparts(here);
addpath(parent);
testCase.TestData.parent = parent;
end


function test_config_values(testCase)
cfg = buildWaveformConfig('verbose', false);

% TS 38.104: 100 MHz @ 30 kHz SCS → 273 RBs (was 66 in the bug)
verifyEqual(testCase, cfg.numRB, 273, ...
    'numRB for 100 MHz/30 kHz SCS must be 273 (TS 38.104).');

% Nsc, Nsym, sample rate
verifyEqual(testCase, cfg.Nsc,  273*12);
verifyEqual(testCase, cfg.Nsym, 14*8);
verifyTrue (testCase, cfg.fs_Hz > cfg.BW_Hz);

% Range resolution = c/(2·BW)
verifyEqual(testCase, cfg.rangeRes_m, 299792458/(2*cfg.BW_Hz), ...
    'AbsTol', 1e-9);

% Max velocity = λ·SCS/4 (corrected formula)
expectedVmax = cfg.lambda_m * cfg.SCS_Hz / 4;
verifyEqual(testCase, cfg.maxVel_mps, expectedVmax, 'AbsTol', 1e-6);

% kTBF noise floor sane (≈ -94 dBm at 100 MHz, 7 dB NF)
expected_dBm = 10*log10(cfg.noisePower_lin*1e3);
verifyTrue(testCase, expected_dBm > -100 && expected_dBm < -85, ...
    'Thermal noise floor outside expected range.');
end


function test_ofdm_roundtrip(testCase)
% demod(mod(grid)) must equal grid to numerical precision.
cfg = buildWaveformConfig('verbose', false, 'numSlots', 1);
[~, info] = isacOFDMWaveform(cfg, RandStream('mt19937ar','Seed',0));

% Modulate a known random grid
Nsc = info.Nsc; Nsym = info.Nsym;
rs   = RandStream('mt19937ar','Seed',7);
grid = (randn(rs,Nsc,Nsym) + 1j*randn(rs,Nsc,Nsym))/sqrt(2);
grid(floor(Nsc/2)+1, :) = 0;          % mimic DC null

tx = localOFDMMod(grid, info.Nfft, info.cpLen, Nsc);
gridHat = localOFDMDemod(tx, info.Nfft, info.cpLen, Nsc, Nsym);

err = max(abs(grid(:) - gridHat(:)));
verifyLessThan(testCase, err, 1e-10, ...
    'OFDM round-trip exceeds numerical precision.');
end


function test_single_target_RD(testCase)
% One synthetic target → peak in RD map matches ground truth.
cfg = buildWaveformConfig('verbose', false);

% Target: known range, known radial velocity
o          = struct();
o.size     = 'medium';
o.sizeIdx  = 2;
o.position = cfg.gNB_pos + [75; 0; -cfg.gNB_pos(3)];   % R=75 m horizontal
o.velocity = [-12; 0; 0];                               % toward gNB
o.rcs      = 20;
o.isMoving = true;

% Run the chain (no thermal noise injection — addNoise sees this scene)
seed = 42;
[rdMap, info] = simulateISAC(o, cfg, struct('seed', seed));

% Find peak
[~, lin] = max(rdMap(:));
[d_pk, r_pk] = ind2sub(size(rdMap), lin);

r_est = info.rangeAxis(r_pk);
v_est = info.velocityAxis(d_pk);

% Ground truth from info.attachGroundTruth (3-D)
R_gt = info.targetRanges_m(1);
v_gt = info.targetVels_mps(1);

% Allow ±2 bins (windowing + interpolation tolerances)
verifyLessThan(testCase, abs(r_est - R_gt), 2*info.rangeBin_m, ...
    sprintf('Range peak off: est=%.2f, gt=%.2f, bin=%.3f', ...
            r_est, R_gt, info.rangeBin_m));

verifyLessThan(testCase, abs(v_est - v_gt), 2*info.velBin_mps, ...
    sprintf('Doppler peak off: est=%.2f, gt=%.2f, bin=%.3f', ...
            v_est, v_gt, info.velBin_mps));
end


function test_empty_scene(testCase)
cfg = buildWaveformConfig('verbose', false);
objects = generateObjects(0);
[rdMap, info] = simulateISAC(objects, cfg, struct('seed', 1));

verifyEqual(testCase, size(rdMap), [cfg.numDopplerBins cfg.numRangeBins]);
verifyTrue (testCase, all(isfinite(rdMap(:))));
verifyEqual(testCase, info.numTargets, 0);
end


function test_size_indices(testCase)
% Ensure every type returned has the correct integer size label.
mapping = containers.Map( ...
    {'small','medium','large'}, ...
    { 1,      2,       3     });
objs = generateObjects(20, struct('sizeMix', [1 1 1]));
for i = 1:numel(objs)
    verifyEqual(testCase, objs(i).sizeIdx, mapping(objs(i).size), ...
        sprintf('Size index mismatch for size ''%s''', objs(i).size));
end
end


function test_dataset_amplitude(testCase)
% Two targets with very different RCS → the higher-RCS target should
% produce the brighter RD-map peak, confirming amplitude is preserved
% (no per-frame normalisation away of RCS information).
cfg = buildWaveformConfig('verbose', false);

oBig = struct('size','large','sizeIdx',3, ...
    'position',cfg.gNB_pos+[120;-20;-cfg.gNB_pos(3)], ...
    'velocity',[0;0;0],'rcs',1000,'isMoving',false);

oSmall = struct('size','small','sizeIdx',1, ...
    'position',cfg.gNB_pos+[80;25;-cfg.gNB_pos(3)], ...
    'velocity',[3;0;0],'rcs',0.05,'isMoving',true);

objs = [oBig oSmall];
[rd, info] = simulateISAC(objs, cfg, struct('seed', 99));

% Sample RD map at the (range, velocity) bin of each target
[~, ridx_big]   = min(abs(info.rangeAxis    - info.targetRanges_m(1)));
[~, didx_big]   = min(abs(info.velocityAxis - info.targetVels_mps(1)));
[~, ridx_small] = min(abs(info.rangeAxis    - info.targetRanges_m(2)));
[~, didx_small] = min(abs(info.velocityAxis - info.targetVels_mps(2)));

amp_big   = rd(didx_big,   ridx_big);
amp_small = rd(didx_small, ridx_small);

% The 1000:0.05 RCS ratio (43 dB) should dominate any 1/R^4 disadvantage
% of the more distant building (120 vs 80 m → 7 dB) so amp_big > amp_small.
verifyGreaterThan(testCase, amp_big, amp_small, ...
    'High-RCS target should produce brighter RD peak.');
end


% ── helpers (mirror the modulator in isacOFDMWaveform.m) ─────────────
function tx = localOFDMMod(grid, Nfft, cpLen, Nsc)
[~, Nsym] = size(grid);
halfLow  = floor(Nsc/2);
halfHigh = Nsc - halfLow;
tx = zeros((Nfft+cpLen)*Nsym, 1);
for s = 1:Nsym
    f = zeros(Nfft, 1);
    f(Nfft-halfLow+1:Nfft) = grid(1:halfLow, s);
    f(1:halfHigh)          = grid(halfLow+1:Nsc, s);
    t = ifft(f, Nfft) * sqrt(Nfft);
    tx((s-1)*(Nfft+cpLen)+(1:(Nfft+cpLen))) = [t(end-cpLen+1:end); t];
end
end

function g = localOFDMDemod(sig, Nfft, cpLen, Nsc, Nsym)
g = complex(zeros(Nsc, Nsym));
halfLow  = floor(Nsc/2);
halfHigh = Nsc - halfLow;
symLen = Nfft + cpLen;
for s = 1:Nsym
    idx = (s-1)*symLen + 1;
    if idx + symLen - 1 > length(sig); break; end
    t = sig(idx+cpLen : idx+symLen-1);
    F = fft(t, Nfft) / sqrt(Nfft);
    g(1:halfLow,    s) = F(Nfft-halfLow+1 : end);
    g(halfLow+1:Nsc,s) = F(1 : halfHigh);
end
end
