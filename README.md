# UMCF

Unified Multimodal Coupled Fingerprint.

## Model

Input `x = (x_s, x_v, x_f)`.

| Layer | Symbol | Dim | Content |
| --- | --- | --- | --- |
| Static | `z_s` | 10 | size, area, area-to-mass ratio, shape complexity, visible reflectance, IR emissivity, solar absorptance, roughness, permittivity, log10 conductivity |
| Slowly varying | `z_v` | 10 | rates of a, e, i, RAAN, argp, n; period; nodal drift; perigee drift; aging |
| Fast-varying | `z_f` | 16 | GI on 64-sample optical, radar, IR, polarization, and spectral sequences |
| Coupling | `z_c` | 20 | PCA of vec(z_s z_v^T), vec(z_s z_f^T), vec(z_v z_f^T) (420-D) |


## Metrics

5 classes x 2000 samples, split 7:3, 20 repeats, mean +/- std. Inner 10% of the training split selects `tau` and `d_c`.

- FDR = tr(S_B) / (tr(S_W) + eps)
- Rank-1 = N_correct / N_query
- Mean and minimum inter-class distance
- Mean and maximum intra-class drift
- Geometric-perturbation drift (5 deg attitude and solar-phase jitter before GI)
- R_noise = R1(SNR) / R1(20 dB), SNR 0, 5, ..., 30
- R_miss = FDR(rho) / FDR(0), missing ratio 0 to 0.6

Baseline: 20 dB, no missing. Ablation noise: 10 dB, no missing. Ablation missing: 20 dB, 30% missing. Modules: GI, CE, Fisher.

Released-code check: seed 42+r, train-only fit, test clipped to the training min/max. Variants: static removed; size, area, and area-to-mass ratio only; orbit block plus 8 optical and radar statistics. Reference values: `reference/published_released_check.json`.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10, PyTorch 1.13.

## Run

```bash
python -m umcf generate --out data/processed/umcf_semireal.npz
python -m umcf train --data data/processed/umcf_semireal.npz --out checkpoints/umcf.pt
python -m umcf test --data data/processed/umcf_semireal.npz --ckpt checkpoints/umcf.pt --out results/umcf_metrics.json
python -m umcf infer --data data/processed/umcf_semireal.npz --ckpt checkpoints/umcf.pt --index 0 --out results/infer.json
python -m umcf experiment --data data/processed/umcf_semireal.npz --repeats 20 --baselines --ablation --out results
python -m umcf released-check --data data/processed/umcf_semireal.npz --out results/released_check.json
python tests/test_equations.py
```

`--quick` reduces the sample count and the number of repeats.

## Layout

```
configs/default.yaml
umcf/model.py gi.py fisher.py distance.py metrics.py
umcf/data.py physics.py orbit.py prototypes.py
umcf/train.py test.py infer.py generate.py
umcf/baselines.py experiment.py released_check.py
scripts/ tests/test_equations.py requirements.txt
```
