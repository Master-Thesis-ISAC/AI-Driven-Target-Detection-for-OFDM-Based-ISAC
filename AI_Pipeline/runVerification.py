"""
run_verification.py
Verifies all 5 Python pipeline files are working correctly.
Run from the ai_pipeline/ folder:
    python run_verification.py

Does NOT require a real dataset - all tests use synthetic data.
"""
import sys
import traceback
import numpy as np
import torch

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    print(f"[{'':2}] {name} ... ", end="", flush=True)
    try:
        fn()
        print("PASS")
        passed += 1
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        failed += 1


# ------------------------------------------------------------------ #
# TEST 1: dataset normalisation
# ------------------------------------------------------------------ #
def t_normalise():
    from dataset import normalise_rdmap_db, RDMAP_OFFSET_DB, RDMAP_SCALE_DB
    rd = np.array([[RDMAP_OFFSET_DB]], dtype=np.float32)
    out = normalise_rdmap_db(rd)
    assert abs(float(out[0, 0])) < 1e-6, "Centre value should normalise to 0"

    rd2 = np.array([[RDMAP_OFFSET_DB + RDMAP_SCALE_DB]], dtype=np.float32)
    out2 = normalise_rdmap_db(rd2)
    assert abs(float(out2[0, 0]) - 1.0) < 1e-6, "Offset+scale should normalise to 1"

test("dataset.normalise_rdmap_db", t_normalise)


# ------------------------------------------------------------------ #
# TEST 2: load_scenario_mat shape check (synthetic HDF5 file)
# ------------------------------------------------------------------ #
def t_load_mat():
    import tempfile, os, h5py
    from dataset import load_scenario_mat

    tmp = tempfile.mktemp(suffix=".mat")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("rdMap_dB",         data=np.zeros((64, 128), dtype=np.float32))
        f.create_dataset("targetPresent",    data=np.array([1]))
        f.create_dataset("sceneLargestSize", data=np.array([2]))

    s = load_scenario_mat(tmp)
    os.remove(tmp)
    assert s["rdMap_dB"].shape == (64, 128), "Wrong RD map shape"
    assert s["targetPresent"]    == 1,       "Wrong presence label"
    assert s["sceneLargestSize"] == 2,       "Wrong size label"

test("dataset.load_scenario_mat", t_load_mat)


# ------------------------------------------------------------------ #
# TEST 3: ISACDataset with a synthetic manifest + mat files
# ------------------------------------------------------------------ #
def t_dataset():
    import tempfile, os, h5py, pandas as pd
    from dataset import ISACDataset

    tmp_dir = tempfile.mkdtemp()
    files = []
    rows  = []
    for i in range(4):
        fname = os.path.join(tmp_dir, f"scenario_{i:04d}.mat")
        with h5py.File(fname, "w") as f:
            f.create_dataset("rdMap_dB",         data=np.random.randn(64, 128).astype(np.float32))
            f.create_dataset("targetPresent",    data=np.array([i % 2]))
            f.create_dataset("sceneLargestSize", data=np.array([i % 4]))
        files.append(fname)
        rows.append({"idx": i, "type": "test", "numObj": i%2,
                     "targetPresent": i%2, "sceneLargestSize": i%4,
                     "SNR_sense_dB": 10, "SNR_comm_dB": 15,
                     "BER": 0.01, "EVM_dB": -20,
                     "CFAR_present": 0, "Energy_present": 0,
                     "UE_x": 100, "UE_y": 20,
                     "seed": i, "split": "train", "file": fname})

    manifest = os.path.join(tmp_dir, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest, index=False)

    ds = ISACDataset(manifest, split="train", augment=True)
    assert len(ds) == 4, "Wrong dataset length"

    img, pres, size = ds[0]
    assert img.shape   == (1, 64, 128), f"Wrong image shape: {img.shape}"
    assert pres.dtype  == torch.float32, "Presence should be float32"
    assert size.dtype  == torch.long,    "Size should be long"

    # cleanup
    import shutil
    shutil.rmtree(tmp_dir)

test("dataset.ISACDataset", t_dataset)


# ------------------------------------------------------------------ #
# TEST 4: model forward pass shapes
# ------------------------------------------------------------------ #
def t_model_forward():
    from model import ISACDetectorCNN, count_parameters
    model = ISACDetectorCNN()
    assert count_parameters(model) > 0, "No parameters"

    x = torch.randn(4, 1, 64, 128)
    p, s = model(x)
    assert p.shape == (4,),    f"Presence shape wrong: {p.shape}"
    assert s.shape == (4, 4),  f"Size shape wrong: {s.shape}"
    assert all(torch.isfinite(p)), "Non-finite presence logits"
    assert all(torch.isfinite(s).all(dim=-1)), "Non-finite size logits"

test("model.ISACDetectorCNN forward pass", t_model_forward)


# ------------------------------------------------------------------ #
# TEST 5: model output types
# ------------------------------------------------------------------ #
def t_model_outputs():
    from model import ISACDetectorCNN
    model = ISACDetectorCNN()
    model.eval()
    x = torch.randn(2, 1, 64, 128)
    with torch.no_grad():
        p, s = model(x)
    probs = torch.sigmoid(p)
    assert (probs >= 0).all() and (probs <= 1).all(), "Sigmoid output out of [0,1]"
    preds = s.argmax(dim=-1)
    assert ((preds >= 0) & (preds <= 3)).all(), "Size class out of [0,3]"

test("model.ISACDetectorCNN output range", t_model_outputs)


# ------------------------------------------------------------------ #
# TEST 6: training step (one mini-batch, no real data)
# ------------------------------------------------------------------ #
def t_train_step():
    from model import ISACDetectorCNN
    import torch.nn as nn
    from torch.optim import Adam

    model = ISACDetectorCNN()
    opt   = Adam(model.parameters(), lr=1e-3)
    bce   = nn.BCEWithLogitsLoss()
    ce    = nn.CrossEntropyLoss()

    x      = torch.randn(8, 1, 64, 128)
    y_pres = torch.randint(0, 2, (8,)).float()
    y_size = torch.randint(0, 4, (8,))

    params_before = [p.clone() for p in model.parameters()]
    opt.zero_grad()
    p_logit, s_logits = model(x)
    loss = bce(p_logit, y_pres) + 0.5 * ce(s_logits, y_size)
    loss.backward()
    opt.step()
    params_after = list(model.parameters())

    changed = any(not torch.equal(pb, pa)
                  for pb, pa in zip(params_before, params_after))
    assert changed,           "Parameters did not update after backward"
    assert torch.isfinite(loss), "Loss is not finite"

test("train: single gradient step", t_train_step)


# ------------------------------------------------------------------ #
# TEST 7: collect_predictions (evaluate.py)
# ------------------------------------------------------------------ #
def t_collect_predictions():
    from model import ISACDetectorCNN
    from evaluate import collect_predictions
    from torch.utils.data import DataLoader, TensorDataset

    model = ISACDetectorCNN()
    model.eval()

    x      = torch.randn(12, 1, 64, 128)
    y_pres = torch.randint(0, 2, (12,)).float()
    y_size = torch.randint(0, 4, (12,))
    loader = DataLoader(TensorDataset(x, y_pres, y_size), batch_size=4)

    pred = collect_predictions(model, loader, torch.device("cpu"))
    assert len(pred["pres_score"]) == 12, "Wrong number of scores"
    assert len(pred["size_pred"])  == 12, "Wrong number of size preds"
    assert ((pred["pres_score"] >= 0) & (pred["pres_score"] <= 1)).all(), \
        "Scores outside [0,1]"
    assert ((pred["size_pred"] >= 0) & (pred["size_pred"] <= 3)).all(), \
        "Size preds outside [0,3]"

test("evaluate.collect_predictions", t_collect_predictions)


# ------------------------------------------------------------------ #
# TEST 8: dataset.normalise (augmentation does not change shape)
# ------------------------------------------------------------------ #
def t_augment_shape():
    from dataset import ISACDataset
    import tempfile, os, h5py, pandas as pd, shutil

    tmp_dir = tempfile.mkdtemp()
    fname   = os.path.join(tmp_dir, "scenario_0000.mat")
    with h5py.File(fname, "w") as f:
        f.create_dataset("rdMap_dB",         data=np.zeros((64,128), dtype=np.float32))
        f.create_dataset("targetPresent",    data=np.array([1]))
        f.create_dataset("sceneLargestSize", data=np.array([1]))

    manifest = os.path.join(tmp_dir, "manifest.csv")
    pd.DataFrame([{"idx":0,"type":"t","numObj":1,"targetPresent":1,
                   "sceneLargestSize":1,"SNR_sense_dB":10,"SNR_comm_dB":15,
                   "BER":0.01,"EVM_dB":-20,"CFAR_present":1,"Energy_present":1,
                   "UE_x":100,"UE_y":20,"seed":0,"split":"train","file":fname}
                  ]).to_csv(manifest, index=False)

    ds = ISACDataset(manifest, split="train", augment=True)
    for _ in range(10):
        img, _, _ = ds[0]
        assert img.shape == (1, 64, 128), f"Augmentation changed shape to {img.shape}"

    shutil.rmtree(tmp_dir)

test("dataset: augmentation preserves shape", t_augment_shape)


# ------------------------------------------------------------------ #
# TEST 9: reconstruct_scene.cluster_cfar_peaks
# ------------------------------------------------------------------ #
def t_cluster():
    from reconstruct_scene import cluster_cfar_peaks

    # Empty input
    c = cluster_cfar_peaks(np.zeros((0, 2), dtype=int))
    assert len(c) == 0, "Empty input should give empty output"

    # Two nearby peaks -> should merge to 1 centroid
    peaks = np.array([[60, 30], [61, 31], [62, 30]])
    c = cluster_cfar_peaks(peaks, min_distance=3)
    assert len(c) == 1, f"3 nearby peaks should cluster to 1, got {len(c)}"

    # Two well-separated peaks -> should stay as 2
    peaks2 = np.array([[10, 10], [100, 50]])
    c2 = cluster_cfar_peaks(peaks2, min_distance=3)
    assert len(c2) == 2, f"2 separated peaks should stay 2, got {len(c2)}"

test("reconstruct_scene.cluster_cfar_peaks", t_cluster)


# ------------------------------------------------------------------ #
# TEST 10: reconstruct_scene.predict_scene (synthetic mat)
# ------------------------------------------------------------------ #
def t_predict_scene():
    import tempfile, os, h5py, shutil
    from model import ISACDetectorCNN
    from reconstruct_scene import load_full_mat, predict_scene

    tmp_dir = tempfile.mkdtemp()
    fname   = os.path.join(tmp_dir, "scenario_0001.mat")

    range_axis    = np.linspace(0, 192, 128).astype(np.float64)
    velocity_axis = np.linspace(-184, 178, 64).astype(np.float64)
    rd_db = np.zeros((64, 128), dtype=np.float32)
    rd_db[32, 60] = 40.0          # artificial target peak

    with h5py.File(fname, "w") as f:
        f.create_dataset("rdMap_dB",             data=rd_db)
        f.create_dataset("targetPresent",        data=np.array([1]))
        f.create_dataset("sceneLargestSize",     data=np.array([2]))
        f.create_dataset("rangeAxis",            data=range_axis)
        f.create_dataset("velocityAxis",         data=velocity_axis)
        f.create_dataset("targetRange_m",        data=np.array([90.0]))
        f.create_dataset("targetVel_mps",        data=np.array([0.0]))
        f.create_dataset("targetSizeIdx",        data=np.array([2]))
        f.create_dataset("classicalCFAR_peaks",  data=np.array([[61, 33]]))

    full = load_full_mat(fname)
    assert full["rdMap_dB"].shape == (64, 128), "RD map shape wrong"
    assert full["cfarPeaks_rdBin"].shape[1] == 2, "CFAR peaks shape wrong"

    model = ISACDetectorCNN()
    model.eval()
    pred = predict_scene(model, full, torch.device("cpu"))

    assert "presence_prob"   in pred, "Missing presence_prob"
    assert "scene_size_name" in pred, "Missing scene_size_name"
    assert "targets"         in pred, "Missing targets"
    assert 0.0 <= pred["presence_prob"] <= 1.0, "presence_prob out of range"

    shutil.rmtree(tmp_dir)

test("reconstruct_scene.predict_scene", t_predict_scene)


# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
print(f"\n{'='*50}")
print(f"Results: {passed} passed,  {failed} failed  (of {passed+failed} tests)")
if failed == 0:
    print("All tests passed. Safe to push.")
else:
    print("Fix failing tests before pushing.")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)