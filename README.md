# AI-Driven Target Detection for OFDM-Based ISAC Systems

**Master of Science Thesis — Telecommunication Systems**  
Blekinge Institute of Technology (BTH), May 2026

**Authors:** Raghu Vamsi Sai Rangannagari · Omkar Siddeswar Tenali  
**Supervisor:** Prof. Dr.-Ing. Hans-Jürgen Zepernick

---

## Overview

This repository contains the full simulation and AI pipeline for the thesis:

> *AI-Driven Target Detection for OFDM-Based Integrated Sensing and Communication Systems — A Simulation Study Using 5G NR Waveforms*

The work investigates whether a small Convolutional Neural Network (CNN) can outperform the classical CA-CFAR detector for target presence detection in an OFDM-based ISAC system, and whether the same model can additionally classify targets by size. A 5G NR-compliant MATLAB simulator generates a labelled dataset of Range-Doppler maps, which a two-head CNN is then trained and evaluated on.

**Key results:**
- CNN presence detection accuracy: **99.86%** (AUC = 1.000)
- CNN detection rate Pd: **99.71%** vs CA-CFAR: **92.71%** at equal Pfa = 0
- CNN four-class size classification accuracy: **84.10%**
- Joint sensing does not degrade communication BER

---

## Repository Structure

```
Final_Master_Thesis/
├── README.md
├── .gitignore
├── IMPLEMENTATION_NOTES.md
├── Simulator/                  MATLAB dataset generation pipeline
│   ├── GenerateISACData_Main.m
│   ├── buildWaveformConfig.m
│   ├── generateObjects.m
│   ├── simulateMovement.m
│   ├── simulateISAC.m
│   ├── isacOFDMWaveform.m
│   ├── applySVDPrecoding.m
│   ├── h38901ISACChannel.m
│   ├── addNoise.m
│   ├── isacSensingProcessor.m
│   ├── classicalDetector.m
│   ├── ueDecoder.m
│   ├── saveData.m
│   ├── HARQEntity.m
│   ├── viewScenario.m
│   ├── visualizeScenario.m
│   ├── evaluateDataset.m
│   ├── generate_RQ3_BER_EVM.m
│   ├── run_verification.m
│   └── tests/
│       └── test_isac_pipeline.m
└── AI_Pipeline/                Python training and evaluation pipeline
    ├── dataset.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    ├── reconstruct_scene.py
    ├── requirements.txt
    └── runVerification.py
```

---

## Requirements

### MATLAB (Simulator)
- MATLAB R2022b or later
- 5G Toolbox (required for `nrCDLChannel`; the simulator falls back to an internal CDL model if absent)
- Signal Processing Toolbox (for `taylorwin`; falls back to `hann` if absent)

### Python (AI Pipeline)
- Python 3.9 or later
- Dependencies listed in `AI_Pipeline/requirements.txt`

Install Python dependencies:
```bash
cd AI_Pipeline
pip install -r requirements.txt
```

---

## Execution Order

### Step 1 — Generate the dataset (MATLAB)

```matlab
cd Simulator
GenerateISACData_Main
```

This generates 5000 labelled scenarios into an `output/` folder and writes `output/manifest.csv`. Each scenario is saved as a `scenario_XXXX.mat` file containing the RD map, ground-truth labels, CFAR detector outputs, and BER/EVM metrics.

Configuration (edit at the top of `GenerateISACData_Main.m`):
- `numScenariosEach` — scenarios per template (default 500)
- `outputDir` — output folder path
- `doVisualize` — set `true` to show plots during generation

### Step 2 — Train the CNN (Python)

```bash
cd AI_Pipeline
python train.py --manifest ../output/manifest.csv \
                --epochs 30 \
                --batch_size 64 \
                --out_dir checkpoints
```

Saves `checkpoints/best_model.pt` (best validation loss) and `checkpoints/training_log.csv`.

### Step 3 — Evaluate on the test set (Python)

```bash
python evaluate.py --manifest ../output/manifest.csv \
                   --checkpoint checkpoints/best_model.pt \
                   --out_dir results
```

Produces the five thesis figures and `results/summary.csv`:
- `RQ1_AI_confusion_matrix.png`
- `RQ2_ROC_AI_vs_CFAR.png`
- `RQ2_Pd_vs_SNR.png`
- `RQ2_size_confusion_matrix.png`
- `RQ2_size_acc_vs_SNR.png`

### Step 4 — Scene reconstruction (Python)

```bash
python reconstruct_scene.py --manifest ../output/manifest.csv \
                            --checkpoint checkpoints/best_model.pt \
                            --num_examples 6
```

### Step 5 — BER/EVM figure (MATLAB)

```matlab
cd Simulator
generate_RQ3_BER_EVM
```

Reads `output/manifest.csv` and saves `RQ3_BER_vs_SNR_with_theory.png`.

---

## Dataset

The full 5000-scenario dataset (297 MB) is not included in this repository due to file size.

**Download:** [ISAC\_Dataset\_5000\_Scenarios.zip — Google Drive](https://drive.google.com/file/d/1fhyPWFJJGfA9q4Q9eCBDjz-Akze4uE8m/view?usp=sharing)

After downloading, extract into the `output/` folder:

```
Final_Master_Thesis/
└── output/
    ├── manifest.csv
    ├── scenario_0001.mat
    ├── scenario_0002.mat
    └── ...
```

Then run the AI pipeline pointing to `output/manifest.csv`.

---

## Verification

Both pipelines include standalone verification scripts that do not require the full dataset.

**MATLAB:**
```matlab
cd Simulator
run_verification        % 14 unit tests
cd tests
runtests('test_isac_pipeline')   % 6 physics/signal-processing tests
```

**Python:**
```bash
cd AI_Pipeline
python runVerification.py   % 10 unit tests
```

All 30 tests pass on the development machine (MATLAB R2026a, Python 3.11, PyTorch 2.0, Windows 11).

---

## Simulator Design

The MATLAB pipeline follows 3GPP standards throughout:

| Component | Standard |
|---|---|
| OFDM numerology (100 MHz, 30 kHz SCS, 273 RBs) | 3GPP TS 38.104, TS 38.211 |
| CDL-B channel model | 3GPP TR 38.901 |
| DMRS pilot pattern | 3GPP TS 38.211 Section 7.4.1.1 |

The sensing processor follows the reciprocal-filtering approach of Sturm & Wiesbeck (2011): divide the received resource grid by the transmitted symbols, then apply a 2D windowed FFT to obtain the Range-Doppler map.

---

## CNN Architecture

```
Input: (1, 64, 128) RD map
  └─ Conv-BN-ReLU-MaxPool  ×3   →  (128, 8, 16)
  └─ AdaptiveAvgPool2d(1)        →  (128,)
  └─ Linear(128→64) + ReLU + Dropout(0.5)
        ├─ head_presence: Linear(64→1)   →  presence logit
        └─ head_size:     Linear(64→4)   →  size logits
```

Total trainable parameters: ~101,000  
Loss: `BCE(presence) + 0.5 × CrossEntropy(size)`  
Optimiser: Adam, lr=1e-3, ReduceLROnPlateau, early stopping (patience=8)

---

## Citation

If you use this code or dataset, please cite the thesis:

```
Raghu Vamsi Sai Rangannagari and Omkar Siddeswar Tenali,
"AI-Driven Target Detection for OFDM-Based Integrated Sensing and
Communication Systems: A Simulation Study Using 5G NR Waveforms",
Master of Science Thesis, Blekinge Institute of Technology, May 2026.
```

---

## License

This repository is released for academic use. See `LICENSE` for details.
