function [txWaveform, waveformInfo] = isacOFDMWaveform(cfg, rngStream)
% isacOFDMWaveform  Generate a 5G NR OFDM transmit waveform.
%
% Builds the resource grid, inserts DMRS pilots and QAM data symbols,
% nulls the DC subcarrier, then runs OFDM modulation with constant CP.
% DMRS positions follow TS 38.211 Type-1 CDM-0: symbols 2,7,8,11 per slot.
%
% Inputs:
%   cfg       : struct from buildWaveformConfig
%   rngStream : RandStream for per-scenario data randomness (optional)

if nargin < 2 || isempty(rngStream)
    rngStream = RandStream('mt19937ar', 'Seed', 'shuffle');
end

Nsc      = cfg.Nsc;
Nsym     = cfg.Nsym;
Nfft     = cfg.Nfft;
cpLen    = cfg.cpLen;
modOrder = cfg.modOrder;

% Resource grid
grid = complex(zeros(Nsc, Nsym));

% DMRS pilot mask: every other subcarrier at symbols 2,7,8,11 of each slot
pilotSymsInSlot = [2 7 8 11];
pilotMask = false(Nsc, Nsym);
for sl = 0:cfg.numSlots-1
    for ls = pilotSymsInSlot
        symIdx = sl*cfg.numSymbolsPerSlot + ls + 1;
        if symIdx > Nsym; continue; end
        pilotMask(1:2:end, symIdx) = true;
    end
end

% Pilots use a fixed RandStream so the receiver can regenerate them
pilotStream = RandStream('mt19937ar', 'Seed', 1);
nPilots     = sum(pilotMask(:));
pilotBits   = randi(pilotStream, [0 1], nPilots, 2);
grid(pilotMask) = ((1 - 2*pilotBits(:,1)) + 1j*(1 - 2*pilotBits(:,2))) / sqrt(2);

% Data symbols use per-scenario rngStream
dataMask = ~pilotMask;
nData    = sum(dataMask(:));
nBits    = nData * cfg.bitsPerSym;
dataBits = randi(rngStream, [0 1], nBits, 1);
dataSyms = qammod(dataBits, modOrder, 'InputType', 'bit', 'UnitAveragePower', true);
grid(dataMask) = dataSyms;

% Null the DC subcarrier (5G NR convention)
dcSCIdx = floor(Nsc/2) + 1;
grid(dcSCIdx, :) = 0;

% OFDM modulation with constant CP
txWaveform = ofdmModulate(grid, Nfft, cpLen, Nsc);

% Normalise to unit average power
pwr = mean(abs(txWaveform).^2);
if pwr > 0
    txWaveform = txWaveform / sqrt(pwr);
end

% Output info struct
waveformInfo             = struct();
waveformInfo.resourceGrid    = grid;
waveformInfo.pilotMask       = pilotMask;
waveformInfo.dataMask        = dataMask;
waveformInfo.Nsc             = Nsc;
waveformInfo.Nsym            = Nsym;
waveformInfo.Nfft            = Nfft;
waveformInfo.cpLen           = cpLen;
waveformInfo.fs_Hz           = cfg.fs_Hz;
waveformInfo.SCS_Hz          = cfg.SCS_Hz;
waveformInfo.subcarrierFreqs = cfg.SCS_Hz * ((-floor(Nsc/2)):(ceil(Nsc/2)-1)).';
waveformInfo.numSlots        = cfg.numSlots;
waveformInfo.modOrder        = modOrder;
waveformInfo.dataBits        = dataBits;
end


function tx = ofdmModulate(grid, Nfft, cpLen, Nsc)
% Internal OFDM modulator. Subcarrier mapping:
%   grid rows 1..floor(Nsc/2)     -> IFFT bins (Nfft-halfLow+1)..Nfft  (neg freqs)
%   grid rows floor(Nsc/2)+1..Nsc -> IFFT bins 1..ceil(Nsc/2)           (pos freqs)
[Nsc_, Nsym] = size(grid);
assert(Nsc_ == Nsc, 'Grid row count mismatch.');
halfLow  = floor(Nsc/2);
halfHigh = Nsc - halfLow;

tx = zeros((Nfft + cpLen) * Nsym, 1);
for s = 1:Nsym
    f = zeros(Nfft, 1);
    f(Nfft-halfLow+1 : Nfft) = grid(1:halfLow, s);
    f(1 : halfHigh)           = grid(halfLow+1 : Nsc, s);
    timeSym = ifft(f, Nfft) * sqrt(Nfft);
    cp      = [timeSym(end-cpLen+1:end); timeSym];
    idx     = (s-1)*(Nfft+cpLen) + (1:(Nfft+cpLen));
    tx(idx) = cp;
end
end