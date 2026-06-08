function [rxSensing, H_comm, channelMeta] = h38901ISACChannel(...
                                            txMulti, waveformInfo, objects, cfg, mode)
% h38901ISACChannel  3GPP TR 38.901 ISAC channel model.
%
% Computes two outputs from the same transmitted waveform:
%   (a) Comm path  : gNB -> CDL-B multipath -> UE, returns H_comm
%   (b) Sensing path: gNB -> target echoes -> gNB Rx array, returns rxSensing
%
% Echo amplitude follows the radar equation: |alpha|^2 = K * sigma / R^4
% where K is calibrated so a reference target (10 m^2 at 100 m) lands at
% cfg.SNR_sense_dB above the noise floor.
%
% mode: 'full' (default) returns both; 'comm' returns H_comm only (used
%       by simulateISAC for the SVD precoder probe pass).

if nargin < 5 || isempty(mode); mode = 'full'; end

c      = 299792458;
fc     = cfg.fc_Hz;
lambda = cfg.lambda_m;
fs     = waveformInfo.fs_Hz;
Nsamp  = size(txMulti, 1);
Nt     = cfg.Nt;
Nr_c   = cfg.Nr_comm;
Nr_s   = cfg.Nr_sense;
gNB    = cfg.gNB_pos;

% (a) Comm channel
useToolbox = exist('nrCDLChannel', 'file') == 2;
if useToolbox
    try
        H_comm = toolboxCDLMatrix(txMulti, fc, fs, Nt, Nr_c);
    catch
        H_comm = clusteredFlatH(Nt, Nr_c, fc, gNB, cfg.UE_pos, cfg.SNR_comm_dB);
    end
else
    H_comm = clusteredFlatH(Nt, Nr_c, fc, gNB, cfg.UE_pos, cfg.SNR_comm_dB);
end

% (b) Sensing channel
N = numel(objects);

if strcmpi(mode, 'comm')
    rxSensing = complex(zeros(0, Nr_s));
else
    rxSensing = complex(zeros(Nsamp, Nr_s));

    if N > 0
        t_vec  = (0:Nsamp-1).' / fs;
        posMat = reshape([objects.position], 3, N);
        velMat = reshape([objects.velocity], 3, N);
        rcsVec = [objects.rcs];

        rel       = posMat - gNB;
        Rng       = max(sqrt(sum(rel.^2, 1)), 1.0);
        losU      = rel ./ Rng;
        vRad      = sum(velMat .* losU, 1);
        delaySamp = round(2 * Rng / c * fs);
        fDoppler  = 2 * vRad / lambda;

        % Calibrated echo amplitude: reference target (10 m^2, 100 m) at SNR_sense_dB
        SNR_sense_lin = 10^(cfg.SNR_sense_dB / 10);
        K      = SNR_sense_lin * 100^4 / 10;
        amp    = sqrt(K .* rcsVec ./ Rng.^4);

        % Swerling-1 amplitude fluctuation (one realisation per target per CPI)
        amp = amp .* (randn(1,N) + 1j*randn(1,N)) / sqrt(2);

        % Approximate occlusion: attenuate a target by 6 dB if a closer target
        % with >5x its RCS is within 2 degrees azimuth
        if N > 1
            [~, sortIdx] = sort(Rng);
            azDeg = rad2deg(atan2(rel(2,:), rel(1,:)));
            for k = 2:N
                tn = sortIdx(k);
                for m = 1:k-1
                    tm = sortIdx(m);
                    if abs(azDeg(tn) - azDeg(tm)) < 2 && rcsVec(tm) > 5*rcsVec(tn)
                        amp(tn) = amp(tn) * 10^(-6/20);
                    end
                end
            end
        end

        az      = atan2(rel(2,:), rel(1,:));
        rxIdx   = (0:Nr_s-1).';
        rxSteer = exp(1j*pi * rxIdx * sin(az));
        txSum   = sum(txMulti, 2);

        for n = 1:N
            d = delaySamp(n);
            if d >= Nsamp; continue; end
            x_d       = [zeros(d,1); txSum(1:end-d)];
            dopp      = exp(1j*2*pi * fDoppler(n) * t_vec);
            echo      = amp(n) * (x_d .* dopp);
            rxSensing = rxSensing + echo * rxSteer(:,n).';
        end
    end
end

channelMeta = struct( ...
    'fc_Hz',        fc, ...
    'Nt',           Nt, ...
    'Nr_comm',      Nr_c, ...
    'Nr_sense',     Nr_s, ...
    'gNB_pos',      gNB, ...
    'UE_pos',       cfg.UE_pos, ...
    'numTargets',   N, ...
    'channelModel', ternary(useToolbox, 'TR38901_CDL_toolbox', 'TR38901_CDL_internal'));

if N > 0
    channelMeta.targetRanges_m = sqrt(sum((reshape([objects.position],3,N) - gNB).^2, 1));
end
end


function H = toolboxCDLMatrix(txMulti, fc, fs, Nt, Nr)
% CDL-B channel via 5G Toolbox, returns frequency-averaged H matrix.
cdl                           = nrCDLChannel;
cdl.DelayProfile              = 'CDL-B';
cdl.CarrierFrequency          = fc;
cdl.MaximumDopplerShift       = 50;
cdl.SampleRate                = fs;
cdl.TransmitAntennaArray.Size = [Nt 1 1 1 1];
cdl.ReceiveAntennaArray.Size  = [Nr 1 1 1 1];
cdl.NormalizePathGains        = true;

[~, pathGains] = cdl(txMulti);
H = squeeze(mean(mean(pathGains, 1), 2));
H = reshape(H, Nr, Nt);
end


function H = clusteredFlatH(Nt, Nr, fc, gNB, UE, SNR_dB)
% Internal clustered flat-fading channel based on TR 38.901 CDL-B delays/powers.
clusterDelays_ns = [0  20  80  145 195 270];
clusterPow_dB    = [0  -2  -4  -7  -9  -12];
Ncl              = numel(clusterDelays_ns);

d3D    = norm(UE - gNB);
PL_dB  = 32.4 + 21*log10(max(d3D,1)) + 20*log10(fc/1e9);
shadow = 10^(8*randn/10);
PL_lin = 10^(-PL_dB/10) * shadow;
clPow  = 10.^(clusterPow_dB/10);
clPow  = clPow / sum(clPow) * PL_lin * 10^(SNR_dB/10);

H = zeros(Nr, Nt);
for k = 1:Ncl
    AoD = (rand-0.5)*pi;
    AoA = (rand-0.5)*pi;
    txS = exp(1j*pi*(0:Nt-1).'*sin(AoD)) / sqrt(Nt);
    rxS = exp(1j*pi*(0:Nr-1).'*sin(AoA)) / sqrt(Nr);
    g   = (randn + 1j*randn) / sqrt(2);
    H   = H + sqrt(clPow(k)) * (rxS * (g * txS.'));
end
end


function v = ternary(c, a, b); if c; v = a; else; v = b; end; end