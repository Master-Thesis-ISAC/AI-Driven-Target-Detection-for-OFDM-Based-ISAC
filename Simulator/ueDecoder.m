function commMetrics = ueDecoder(X_tx, waveformInfo, H_comm, cfg)
% ueDecoder  UE-side OFDM receiver: channel estimation, equalisation, BER.
%
% Pipeline:
%   1. Apply flat-fading comm channel H_comm and add UE AWGN
%   2. OFDM demodulate each Rx stream
%   3. LS channel estimation from DMRS pilots, interpolate over frequency
%   4. MRC equalisation across Rx streams (per subcarrier)
%   5. QAM demap on data REs, compute BER and EVM

if isempty(H_comm) || any(~isfinite(H_comm(:)))
    commMetrics = struct('BER',NaN,'EVM_dB',NaN,'numBitsTx',0, ...
                         'numBitErrors',0,'SNR_eff_dB',NaN,'H_used',H_comm);
    return;
end

[Nr_c, ~] = size(H_comm);
Nsamp     = size(X_tx, 1);
Nfft      = waveformInfo.Nfft;
cpLen     = waveformInfo.cpLen;
Nsc       = waveformInfo.Nsc;
Nsym      = waveformInfo.Nsym;
modOrder  = waveformInfo.modOrder;
pilotMask = waveformInfo.pilotMask;
dataMask  = waveformInfo.dataMask;
gridTx    = waveformInfo.resourceGrid;

% Step 1: comm channel + UE AWGN
Y_time  = X_tx * H_comm.';
sigPow  = mean(abs(Y_time).^2, 'all');
sigmaSq = max(sigPow / 10^(cfg.SNR_comm_dB/10), eps);
Y_time  = Y_time + sqrt(sigmaSq/2)*(randn(Nsamp,Nr_c)+1j*randn(Nsamp,Nr_c));

% Step 2: OFDM demodulate
Y_freq = complex(zeros(Nsc, Nsym, Nr_c));
for r = 1:Nr_c
    Y_freq(:,:,r) = ofdmDemodCom(Y_time(:,r), Nfft, cpLen, Nsc, Nsym);
end

% Step 3: LS channel estimation from DMRS pilots
H_freq = complex(zeros(Nsc, Nr_c));
for r = 1:Nr_c
    H_pilot = zeros(Nsc, 1);
    counts  = zeros(Nsc, 1);
    for l = 1:Nsym
        col = pilotMask(:, l);
        if ~any(col); continue; end
        Yp    = Y_freq(col, l, r);
        Xp    = gridTx(col, l);
        valid = abs(Xp) > 1e-9;
        rows  = find(col);
        rows  = rows(valid);
        H_pilot(rows) = H_pilot(rows) + Yp(valid) ./ Xp(valid);
        counts(rows)  = counts(rows) + 1;
    end
    valid = counts > 0;
    H_pilot(valid) = H_pilot(valid) ./ counts(valid);
    if any(valid)
        scIdx        = (1:Nsc).';
        H_freq(:, r) = interp1(scIdx(valid), H_pilot(valid), scIdx, 'linear', 'extrap');
    end
end

% Step 4: MRC equalisation
X_hat = complex(zeros(Nsc, Nsym));
denom = sum(abs(H_freq).^2, 2) + 1e-9;
for r = 1:Nr_c
    X_hat = X_hat + conj(H_freq(:,r)) .* Y_freq(:,:,r);
end
X_hat = X_hat ./ denom;

% Step 5: QAM demap on data REs
dcIdx         = floor(Nsc/2) + 1;
useMask       = dataMask;
useMask(dcIdx,:) = false;

dataSymsRx = X_hat(useMask);
dataSymsTx = gridTx(useMask);

bitsRx = qamdemod(dataSymsRx, modOrder, 'OutputType','bit', 'UnitAveragePower',true);
bitsTx = qamdemod(dataSymsTx, modOrder, 'OutputType','bit', 'UnitAveragePower',true);

n         = min(numel(bitsRx), numel(bitsTx));
bitErrors = sum(bitsRx(1:n) ~= bitsTx(1:n));
BER       = bitErrors / max(n, 1);

err_vec = dataSymsRx - dataSymsTx;
sig_pow = mean(abs(dataSymsTx).^2);
err_pow = mean(abs(err_vec).^2);
if sig_pow > 0 && err_pow > 0
    EVM_dB     = 10*log10(err_pow / sig_pow);
    SNR_eff_dB = -EVM_dB;
else
    EVM_dB     = -Inf;
    SNR_eff_dB =  Inf;
end

commMetrics = struct( ...
    'BER',          BER, ...
    'EVM_dB',       EVM_dB, ...
    'numBitsTx',    n, ...
    'numBitErrors', bitErrors, ...
    'SNR_eff_dB',   SNR_eff_dB, ...
    'H_used',       H_comm);
end


function grid = ofdmDemodCom(sig, Nfft, cpLen, Nsc, Nsym)
% OFDM demodulator mirroring isacOFDMWaveform.ofdmModulate.
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