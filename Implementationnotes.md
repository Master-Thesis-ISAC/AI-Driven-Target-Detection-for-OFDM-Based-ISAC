# Implementation Notes and Simulator Validation

This document records the design decisions, validation steps, and corrections
made during the development of the MATLAB ISAC simulator. It serves as a
technical reference for reproducing the dataset and understanding why specific
implementation choices were made.

---

## Standards Alignment

| Parameter | Value | Reference |
|---|---|---|
| Carrier frequency | 3.5 GHz | 3GPP TS 38.104 FR1 |
| Bandwidth | 100 MHz | 3GPP TS 38.104 |
| Subcarrier spacing | 30 kHz | 3GPP TS 38.211 |
| Resource blocks | 273 | 3GPP TS 38.104 Table 5.3.2-1 |
| FFT size | 4096 | Derived from numerology |
| Channel model | CDL-B (NLOS urban) | 3GPP TR 38.901 |
| DMRS pilot positions | Symbols 2, 7, 8, 11 per slot | 3GPP TS 38.211 Section 7.4.1.1 |

---

## Physics and Signal Processing Corrections

### C1. Wrong RB count for 100 MHz / 30 kHz SCS

**File:** `buildWaveformConfig.m`  
**Issue:** The FR1 RB-table row read `100, 30, 66` instead of `100, 30, 273`.  
**Standard:** 3GPP TS 38.104 Table 5.3.2-1 lists 273 RBs for this combination.  
**Impact:** Effective bandwidth was 23.76 MHz instead of 98.28 MHz, giving
range resolution of ~6.3 m instead of ~1.5 m.  
**Fix:** Table corrected to 273 RBs.

### C2. Maximum unambiguous velocity formula wrong by factor Nsym

**File:** `buildWaveformConfig.m`  
**Issue:** `maxVel = lambda * SCS_Hz / (2 * Nsym)` — the Nsym division belongs
in the velocity resolution, not the maximum.  
**Correct:** `vmax = lambda * SCS / 4` (unambiguous Doppler from N pulses with
PRI = T_sym is ±SCS/2, so vmax = ±lambda·SCS/4).  
**Impact:** At fc=3.5 GHz, SCS=30 kHz, Nsym=112 the formula gave ±5.7 m/s.
Every car and drone exceeded this limit and was silently aliased — moving
target labels did not match the RD data at all.  
**Fix:** `maxVel = lambda * SCS_Hz / 4`.

### C3. Inverted subcarrier-to-IFFT mapping; DC subcarrier not nulled

**File:** `isacOFDMWaveform.m`  
**Issue:** Modulator placed upper-half grid at low IFFT bins; demodulator
reversed this. Round trip worked by coincidence but Doppler axis was
sign-mirrored. DC subcarrier was not zeroed (5G NR mandates it null).  
**Impact:** Targets approaching the gNB appeared at positive velocity on the
RD map. DC line spurs corrupted the zero-velocity row.  
**Fix:** Explicit consistent mapping in both modulator and demodulator; DC
subcarrier zeroed before IFFT.

### C4. AWGN added to a magnitude map

**File:** `addNoise.m`  
**Issue:** `noisyMap = abs(rdMap + abs(noise))`. Adding `abs(noise)` (Rayleigh,
strictly positive) to a magnitude map added a positive bias rather than
zero-mean Gaussian noise.  
**Impact:** Noise floor was biased upward; CFAR detectors would not transfer
to real receivers.  
**Fix:** AWGN added as CN(0, sigma^2) to the complex receive waveform before
the sensing processor.

### C5. Echo amplitude formula dimensionally inconsistent

**File:** `h38901ISACChannel.m`  
**Issue:** Echo amplitude conflated power and amplitude scaling and did not
correctly apply the radar equation.  
**Fix:** Amplitude follows the radar equation calibrated so a reference target
(10 m^2 at 100 m) lands at SNR_sense_dB above the noise floor.

### C6. Range axis did not account for zero-padding factor

**File:** `isacSensingProcessor.m`  
**Issue:** Range bin spacing was reported as c/(2·BW) regardless of
zero-pad factor, and `interp2` was used to resample to the AI image size,
distorting peak locations.  
**Fix:** Bin spacing is c/(2·BW·zpFactor). AI image produced by cropping
the FFT output rather than interpolating.

### C7. Stage ordering — precoded waveform not used by channel

**File:** `simulateISAC.m`  
**Issue:** `applySVDPrecoding` output was discarded; the channel function
received the un-precoded waveform. Tx beamforming gain never contributed
to sensing or communication.  
**Fix:** Correct stage ordering with the precoded [Nsamples x Nt] matrix
flowing through the channel.

### C8. HARQ chase combining mathematically incorrect

**File:** `HARQEntity.m`  
**Issue:** Retransmission was modelled as accumulating magnitude copies with
no proper MRC combining. No SNR gain was achievable.  
**Fix:** `chaseCombine` accepts complex pre-detection RD maps and performs MRC.

### C9. Movement step did not match CPI duration

**File:** `simulateMovement.m`  
**Issue:** Hard-coded `dt = 0.01 s` (10 ms). Actual CPI is ~4 ms, so labels
were saved 2.5x ahead of the position seen by the channel.  
**Fix:** Main script passes `dt = cfg.T_CPI_s` derived from the waveform config.

### C10. saveData used 2D range while channel used 3D

**File:** `saveData.m`  
**Issue:** `R = norm(o.position(1:2))` ignored z. Channel used 3D position
with gNB at z=10 m.  
**Impact:** Labels off by ~10 m at close range.  
**Fix:** `R = norm(o.position - cfg.gNB_pos)`.

### C11. RNG reset inside OFDM modulator

**File:** `isacOFDMWaveform.m`  
**Issue:** `rng(42)` at the top of every call produced identical data symbols
on every scenario. The dataset had no per-scenario data randomness.  
**Fix:** Pilots use a fixed sub-stream (so the receiver can regenerate them);
data symbols use a per-scenario RandStream passed in by simulateISAC.

### C12. Per-frame normalisation discarded RCS information

**File:** `saveData.m`  
**Issue:** `normMap = single(rdMap / mapMax)` — every saved frame had peak
amplitude 1.0, destroying the relative brightness between high-RCS and
low-RCS targets.  
**Fix:** RD map saved at absolute amplitude as single; companion rdMap_dB
also written for log-domain consumers.

---

## Dataset Design Decisions

| Decision | Rationale |
|---|---|
| 50/50 present/absent balance | Prevents accuracy from being dominated by majority class |
| 7 sensing SNR levels (0–30 dB) | Covers low-SNR regime where CFAR degrades |
| 5 communication SNR levels (5–25 dB) | Spans from high-BER to zero-BER regime |
| Log-uniform RCS sampling within each size band | Avoids bias toward upper or lower end of each class |
| Scene-level size label = largest target | Consistent with CNN producing one label per RD map |
| Absolute amplitude preserved in saved RD maps | RCS information available for future classification work |
| DMRS pilots on fixed sub-stream, data on per-scenario stream | Receiver can regenerate pilots; each scenario has unique data |
