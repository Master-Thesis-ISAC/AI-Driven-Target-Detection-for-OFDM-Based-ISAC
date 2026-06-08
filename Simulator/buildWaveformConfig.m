function cfg = buildWaveformConfig(varargin)
% buildWaveformConfig  Returns a struct with all 5G NR ISAC waveform parameters.
% Single source of truth for every downstream module.
%
% Usage:
%   cfg = buildWaveformConfig();
%   cfg = buildWaveformConfig('BW_MHz', 100, 'SCS_kHz', 30, ...);
%
% Standards: 3GPP TS 38.104 Table 5.3.2-1, TS 38.211 Section 5.3,
%            TR 38.901 Section 7.5.

p = inputParser;
p.addParameter('fc_GHz',         3.5);
p.addParameter('BW_MHz',         100);
p.addParameter('SCS_kHz',        30);
p.addParameter('modOrder',       16);
p.addParameter('numSlots',       8);
p.addParameter('Nt',             4);
p.addParameter('Nr_comm',        2);
p.addParameter('Nr_sense',       4);
p.addParameter('numLayers',      1);
p.addParameter('SNR_comm_dB',    20);
p.addParameter('SNR_sense_dB',   15);
p.addParameter('numRangeBins',   256);
p.addParameter('numDopplerBins', 128);
p.addParameter('noiseFigure_dB', 7);
p.addParameter('Tnoise_K',       290);
p.addParameter('gNB_pos',        [0;0;10]);
p.addParameter('UE_pos',         [100;30;1.5]);
p.addParameter('verbose',        true);
p.parse(varargin{:});
r = p.Results;

c      = 299792458;
fc_Hz  = r.fc_GHz * 1e9;
BW_Hz  = r.BW_MHz * 1e6;
SCS_Hz = r.SCS_kHz * 1e3;
lambda = c / fc_Hz;

% RB count from TS 38.104 Table 5.3.2-1 (FR1)
rbTable = [ ...
     5,  15,  25;   10, 15,  52;   15, 15,  79;   20, 15, 106; ...
    25,  15, 133;   30, 15, 160;   40, 15, 216;   50, 15, 270; ...
     5,  30,  11;   10, 30,  24;   15, 30,  38;   20, 30,  51; ...
    25,  30,  65;   30, 30,  78;   40, 30, 106;   50, 30, 133; ...
    60,  30, 162;   70, 30, 189;   80, 30, 217;   90, 30, 245; ...
   100,  30, 273; ...
    10,  60,  11;   15, 60,  18;   20, 60,  24; ...
    25,  60,  31;   30, 60,  38;   40, 60,  51;   50, 60,  65; ...
    60,  60,  79;   70, 60,  93;   80, 60, 107;   90, 60, 121; ...
   100,  60, 135];

mask = (rbTable(:,1) == round(BW_Hz/1e6)) & (rbTable(:,2) == r.SCS_kHz);
if any(mask)
    numRB = rbTable(find(mask,1), 3);
else
    numRB = floor(0.95 * BW_Hz / (12 * SCS_Hz));
end

% OFDM grid
Nsc     = numRB * 12;
Nsym    = 14 * r.numSlots;
Nfft    = 2^nextpow2(Nsc);
if Nfft < Nsc * 1.0
    Nfft = 2^(nextpow2(Nsc) + 1);
end
cpLen   = round(0.072 * Nfft);
fs_Hz   = Nfft * SCS_Hz;
T_sym   = 1 / SCS_Hz;
T_symCP = (Nfft + cpLen) / fs_Hz;
T_CPI   = Nsym * T_symCP;

% Sensing resolution and limits
rangeRes = c / (2 * BW_Hz);
maxRange = c / (2 * SCS_Hz);
maxVel   = lambda * SCS_Hz / 4;           % unambiguous velocity: +/-lambda*SCS/4
velRes   = lambda / (2 * Nsym * T_symCP);

% Receiver noise floor
kB           = 1.380649e-23;
noisePowerLin = kB * r.Tnoise_K * BW_Hz * 10^(r.noiseFigure_dB/10);

% Output struct
cfg = struct();
cfg.fc_Hz              = fc_Hz;
cfg.fc_GHz             = r.fc_GHz;
cfg.BW_Hz              = BW_Hz;
cfg.BW_MHz             = r.BW_MHz;
cfg.SCS_Hz             = SCS_Hz;
cfg.SCS_kHz            = r.SCS_kHz;
cfg.lambda_m           = lambda;
cfg.numRB              = numRB;
cfg.Nsc                = Nsc;
cfg.Nsym               = Nsym;
cfg.numSymbolsPerSlot  = 14;
cfg.numSlots           = r.numSlots;
cfg.Nfft               = Nfft;
cfg.cpLen              = cpLen;
cfg.fs_Hz              = fs_Hz;
cfg.T_sym_s            = T_sym;
cfg.T_symCP_s          = T_symCP;
cfg.T_CPI_s            = T_CPI;
cfg.modOrder           = r.modOrder;
cfg.bitsPerSym         = log2(r.modOrder);
cfg.Nt                 = r.Nt;
cfg.Nr_comm            = r.Nr_comm;
cfg.Nr_sense           = r.Nr_sense;
cfg.numLayers          = r.numLayers;
cfg.SNR_comm_dB        = r.SNR_comm_dB;
cfg.SNR_sense_dB       = r.SNR_sense_dB;
cfg.numRangeBins       = r.numRangeBins;
cfg.numDopplerBins     = r.numDopplerBins;
cfg.rangeRes_m         = rangeRes;
cfg.maxRange_m         = maxRange;
cfg.maxVel_mps         = maxVel;
cfg.velRes_mps         = velRes;
cfg.noiseFigure_dB     = r.noiseFigure_dB;
cfg.Tnoise_K           = r.Tnoise_K;
cfg.noisePower_lin     = noisePowerLin;
cfg.gNB_pos            = r.gNB_pos(:);
cfg.UE_pos             = r.UE_pos(:);

cfg.sizeMap = containers.Map( ...
    {'background', 'small', 'medium', 'large'}, ...
    {0, 1, 2, 3});

if r.verbose
    fprintf('[Config] fc=%.2f GHz  BW=%d MHz  SCS=%d kHz  numRB=%d\n', ...
        r.fc_GHz, r.BW_MHz, r.SCS_kHz, numRB);
    fprintf('[Config] Nsc=%d  Nsym=%d  Nfft=%d  CP=%d  fs=%.2f MHz\n', ...
        Nsc, Nsym, Nfft, cpLen, fs_Hz/1e6);
    fprintf('[Config] dr=%.2f m  Rmax=%.0f m  +/-vmax=%.1f m/s  dv=%.2f m/s\n', ...
        rangeRes, maxRange, maxVel, velRes);
    fprintf('[Config] Nt=%d  Nr_comm=%d  Nr_sense=%d  layers=%d  out=[%dx%d]\n', ...
        r.Nt, r.Nr_comm, r.Nr_sense, r.numLayers, ...
        r.numDopplerBins, r.numRangeBins);
end
end