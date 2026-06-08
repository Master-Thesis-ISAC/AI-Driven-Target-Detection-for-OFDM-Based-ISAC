function [rdMap, sensingInfo] = isacSensingProcessor(rxWaveform, txMulti, waveformInfo, cfg)
% isacSensingProcessor  OFDM-radar processing chain (Sturm & Wiesbeck 2011).
%
% Pipeline:
%   1. OFDM demodulate Tx stream and each Rx antenna -> resource grids
%   2. Reciprocal filter: D[k,l] = Y[k,l] / X[k,l]
%   3. Per-antenna 2D windowed FFT, then non-coherent power combining
%   4. Crop to requested [numDopplerBins x numRangeBins] output size

c      = 299792458;
Nsc    = waveformInfo.Nsc;
Nsym   = waveformInfo.Nsym;
Nfft   = waveformInfo.Nfft;
cpLen  = waveformInfo.cpLen;
fs     = waveformInfo.fs_Hz;
SCS_Hz = waveformInfo.SCS_Hz;
fc     = cfg.fc_Hz;
BW     = cfg.BW_Hz;
lambda = cfg.lambda_m;
Nr_ant = size(rxWaveform, 2);
Nr_out = cfg.numRangeBins;
Nd_out = cfg.numDopplerBins;

% Step 1: OFDM demodulation
txStream = sum(txMulti, 2);
X = ofdmDemod(txStream, Nfft, cpLen, Nsc, Nsym);
Y = complex(zeros(Nsc, Nsym, Nr_ant));
for a = 1:Nr_ant
    Y(:,:,a) = ofdmDemod(rxWaveform(:,a), Nfft, cpLen, Nsc, Nsym);
end

% Step 2: Reciprocal filter
threshold = 1e-6 * max(abs(X(:)));
validMask = abs(X) > threshold;
Xinv      = zeros(Nsc, Nsym);
Xinv(validMask) = 1 ./ X(validMask);
D_per_ant = Y .* Xinv;

% Step 3: Per-antenna 2D windowed FFT, non-coherent power combining
Nrange_fft = max(Nr_out, Nsc);
Ndopp_fft  = max(Nd_out, Nsym);

try
    wRange = taylorwin(Nsc,  4, -35);
    wDopp  = taylorwin(Nsym, 4, -35);
catch
    wRange = hann(Nsc);
    wDopp  = hann(Nsym);
end
W2D = wRange * wDopp.';

rdPow = zeros(Ndopp_fft, Nrange_fft);
for a = 1:Nr_ant
    D_a    = D_per_ant(:,:,a) .* W2D;
    RDcube = ifft(D_a, Nrange_fft, 1);
    RDcube = fftshift(fft(RDcube, Ndopp_fft, 2), 2);
    rdPow  = rdPow + abs(RDcube).'.^2;
end
mag = sqrt(rdPow / Nr_ant);

% Step 4: Crop to output size
if Ndopp_fft > Nd_out
    startD = floor((Ndopp_fft - Nd_out)/2) + 1;
    rdMap  = mag(startD:startD+Nd_out-1, 1:Nr_out);
else
    rdMap  = mag(:, 1:Nr_out);
end

% Physical axes
zpRange      = Nrange_fft / Nsc;
rangeRes_bin = c / (2 * BW * zpRange);
rangeAxis    = (0:Nr_out-1) * rangeRes_bin;
maxVel       = lambda * SCS_Hz / 4;
zpDopp       = Ndopp_fft / Nsym;
velRes_bin   = (2*maxVel) / Ndopp_fft;
velocityAxis = (-Nd_out/2 : Nd_out/2-1) * velRes_bin;

sensingInfo = struct( ...
    'rangeAxis',       rangeAxis, ...
    'velocityAxis',    velocityAxis, ...
    'rangeRes_m',      c/(2*BW), ...
    'rangeBin_m',      rangeRes_bin, ...
    'velRes_mps',      lambda/(2*Nsym/SCS_Hz), ...
    'velBin_mps',      velRes_bin, ...
    'maxRange_m',      rangeRes_bin * Nr_out, ...
    'maxVel_mps',      maxVel, ...
    'fc_Hz',           fc, ...
    'BW_Hz',           BW, ...
    'lambda_m',        lambda, ...
    'Nsc',             Nsc, ...
    'Nsym',            Nsym, ...
    'Nrange_fft',      Nrange_fft, ...
    'Ndopp_fft',       Ndopp_fft, ...
    'Nr_ant',          Nr_ant, ...
    'processingChain', 'OFDM_YoverX_2DFFT_noncoherent');
end


function grid = ofdmDemod(sig, Nfft, cpLen, Nsc, Nsym)
% OFDM demodulator: remove CP, FFT, recover Nsc active subcarriers.
% Subcarrier mapping mirrors isacOFDMWaveform.ofdmModulate.
grid     = complex(zeros(Nsc, Nsym));
symLen   = Nfft + cpLen;
halfLow  = floor(Nsc/2);
halfHigh = Nsc - halfLow;

for s = 1:Nsym
    idx = (s-1)*symLen + 1;
    if idx + symLen - 1 > length(sig); break; end
    timeSym = sig(idx + cpLen : idx + symLen - 1);
    F       = fft(timeSym, Nfft) / sqrt(Nfft);
    grid(1:halfLow,     s) = F(Nfft - halfLow + 1 : Nfft);
    grid(halfLow+1:Nsc, s) = F(1 : halfHigh);
end
end