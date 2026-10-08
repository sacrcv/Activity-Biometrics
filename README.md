# ABNet: Activity-Biometrics — Person Identification from Daily Activities

Official PyTorch implementation of [**Activity-Biometrics** (CVPR 2024)](https://arxiv.org/abs/2403.17360),
built on ResNet3D from [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch)
and the [GaitGL](https://github.com/bb12346/GaitGL) architecture.

> Azad & Rawat. *Activity-Biometrics: Person Identification from Daily Activities.* CVPR 2024, pages 287–296.

Person identification from RGB video of **everyday activities**, not just walking.
ABNet attacks the two problems that make this hard — appearance bias and
spatio-temporal complexity — by **disentangling** biometric from non-biometric
features and by **jointly learning activity and identity** so that knowing what
someone is doing helps decide who they are.

| component | mechanism | what it produces | used at inference |
|---|---|---|---|
| video encoder `S_φ` | ResNet3D-50 over 8 frames | `F_AB` | yes |
| actor head `C^B` | transformer decoder + two linear projections | `f_bb`, `f_ba` | yes (`f_bb`) |
| activity head `C^A` | transformer decoder + classifier | `F_Ac` | yes (activity prior) |
| bias-less teacher `T` | GaitGL on binary silhouettes, frozen | `y_T` for `L_KD` | **no** |
| distortion branch `A` | the same weights as `M`, on an elastically distorted clip | `f_bb^D`, `f_ba^D` | **no** |

Retrieval uses `concat(F_Ac, f_bb)`. Silhouettes, the teacher and the distortion
branch are training-only — the paper is explicit that "we only use silhouette
during training and it is not required for inference", and the code is
structured so that this cannot accidentally stop being true.


---

## Install

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate abnet
```

All commands run from the repository root — there is no package to install.
`requirements.txt` gives version ranges; `requirements-lock.txt` pins the exact
versions this was developed and verified against (Python 3.10.20, torch 2.7.1,
CUDA 11.8, A100-40GB).

The only heavy optional dependency is `transformers`, used once, offline, to run
Mask2Former for silhouette extraction. It is never loaded during training or
inference.

### Pretrained weights

| file | size | purpose | where |
|---|---|---|---|
| `r3d50_K_200ep.pth` | ~360 MB | Kinetics-700 ResNet3D-50 init for `S_φ` | [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch#pre-trained-models) |
| `facebook/mask2former-swin-large-coco-instance` | ~850 MB | silhouette extraction | fetched automatically by `tools/extract_silhouettes.py` |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~5 MB | face detection for blurring | fetched automatically by `tools/blur_faces.py` |

Put the ResNet3D weights at `pretrained/r3d50_K_200ep.pth` (the default
`model.pretrained`) or override the path. Training runs without them but warns
loudly: the paper jointly trains an activity head, so Kinetics initialisation is
clearly intended, and a 46M-parameter 3D CNN trained from scratch on 88k clips
will underperform substantially.

GaitGL is trained from scratch here, per dataset, by `train_teacher.py` — no
pretrained gait weights are needed.

> **Note on trained checkpoints.** Trained ABNet model weights are not
> released, as the method is covered by the patent noted at the end of this
> README. Everything needed to train the model from the publicly available
> initialisations above is provided here.


---

## Reproducing the paper

Four stages. The first two are offline preprocessing, then the teacher, then
ABNet itself.

```
videos ──► frames ──► blurred frames ──► silhouettes ──► annotations
                                              │              │
                                              ▼              ▼
                                      GaitGL teacher ──► ABNet
```

### 1. Get the data

Please download the source datasets from their original providers, following
each one's access terms:

| benchmark | derived from | source |
|---|---|---|
| NTU RGB-AB | NTU RGB+D 120 | [rose1.ntu.edu.sg/dataset/actionRecognition](https://rose1.ntu.edu.sg/dataset/actionRecognition/) |
| Charades-AB | Charades | [prior.allenai.org/projects/charades](https://prior.allenai.org/projects/charades) |

Extract frames once:

```bash
python tools/extract_frames.py --videos <path-to-videos> \
    --out data/ntu_rgb_ab/frames --fps 10 --short-side 256
```

### 2. Make it face-restricted

Section 3 of the paper uses a face-restricted setting: "the face of the
individual is blurred so as to avoid learning any of the facial features".

```bash
python tools/blur_faces.py --frames-root data/ntu_rgb_ab/frames
```

This rewrites frames in place (pass `--out` for a copy). Detection misses are
common in these datasets — most NTU subjects are back-facing or distant — so
every frame with no detected face has the top ~22% of the person region blurred
regardless. A missed detection would leak exactly the cue the setting exists to
remove, so the fallback matters more than the detector.

### 3. Extract silhouettes

"The silhouettes of the RGB videos are extracted using Mask2Former [8] to use as
input to `T_θ(·)`."

```bash
python tools/extract_silhouettes.py \
    --frames-root data/ntu_rgb_ab/frames \
    --out data/ntu_rgb_ab/silhouettes
```

Keeps the highest-scoring `person` instance per frame, then applies GaitGL's
pretreatment: crop to the mask, scale by height, and centre horizontally on the
silhouette's **centre of mass** — stable when a limb extends to one side, which a
bbox centre is not. Output is 64×44, GaitGL's native size.

This is the slowest step. Shard it across GPUs:

```bash
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i python tools/extract_silhouettes.py \
    --frames-root data/ntu_rgb_ab/frames --out data/ntu_rgb_ab/silhouettes \
    --shard $i --num-shards 8 &
done; wait
```

`--backend d2` uses the official detectron2 implementation instead of the
HuggingFace one; the weights are identical, so the default avoids a detectron2
build for no loss.

### 4. Build the annotations

Each dataset's **official** train/test split, then a random probe/gallery
division of the test portion (Appendix A). One file per protocol:

```bash
# NTU RGB-AB: official NTU-120 cross-subject split, 26 two-person actions dropped
python tools/prepare_dataset.py --dataset ntu \
    --frames-root data/ntu_rgb_ab/frames \
    --silhouettes-root data/ntu_rgb_ab/silhouettes \
    --out data/ntu_rgb_ab/annotations.json --protocol same_activity

# Charades-AB: official CSVs; `subject` is the actor, `actions` is multi-label
python tools/prepare_dataset.py --dataset charades \
    --charades-csv-dir data/charades_ab/annotations \
    --frames-root data/charades_ab/frames \
    --silhouettes-root data/charades_ab/silhouettes \
    --out data/charades_ab/annotations.json
```

Repeat with `--protocol cross_activity` to get the second annotation file.

Identity labels come straight from the source datasets: NTU encodes the
performer in its directory names (`P###`) and Charades ships a `subject`
column, so neither needs an external mapping. `--dataset manifest` is the
escape hatch for any other benchmark — give it a CSV with `id`, `pid`, `action`
and one of `frames_dir`/`video`.

### 5. Train the teacher (stage 0)

```bash
bash scripts/train_teacher.sh configs/teacher_ntu_rgb_ab.yaml
```

GaitGL learns identities from silhouettes alone. It is "bias-less" because a
binary mask cannot carry clothing color or background, which is what makes it a
useful target for pushing appearance out of `f_bb`.

Each dataset needs its own teacher: `L_KD` is a per-sample KL over the identity
label space, so teacher and student must share that space.

### 6. Train ABNet (stage 1)

```bash
# single GPU
python train.py --cfg configs/ntu_rgb_ab.yaml

# multi-GPU
torchrun --nproc_per_node=8 train.py --cfg configs/ntu_rgb_ab.yaml

# wrapper, GPU count from CUDA_VISIBLE_DEVICES
bash scripts/train.sh configs/ntu_rgb_ab.yaml

# inline overrides
bash scripts/train.sh configs/ntu_rgb_ab.yaml run.batch_size=16 run.max_epoch=30
```

The configs carry the paper's hyperparameters: 8 frames at stride 4, resized to
256×128; batch 32 as 8 identities × 4 clips; Adam at lr 3.5e-4 with weight decay
5e-4; 150 epochs with the LR decayed by 0.1 every 40; triplet margin 0.3;
distortion α = 250; all three loss weights λᵢ = 0.01.


### 7. Evaluate

```bash
bash scripts/test.sh configs/ntu_rgb_ab.yaml output/ntu_rgb_ab/checkpoint_best.pth
```

| protocol | gallery filter | datasets |
|---|---|---|
| `same_activity_include_view` | none | NTU RGB-AB |
| `same_activity_exclude_view` | drop the probe's own camera | NTU RGB-AB |
| `cross_activity_include_view` | drop the probe's activity | NTU RGB-AB |
| `cross_activity_exclude_view` | both | NTU RGB-AB |
| `same_activity` / `cross_activity` | none / activity | Charades-AB (no viewpoint data) |

Reports rank-1, rank-5, rank-10, rank-20, mAP and TAR @ 0.1% FAR. The distance
matrix is computed once and reused across protocols, since they differ only in
which gallery entries are admitted.

---

## Repository layout

```
configs/
  _base.yaml              every default, annotated [paper] or [choice]
  ntu_rgb_ab.yaml         per-dataset overrides and protocol lists
  charades_ab.yaml
  teacher_*.yaml          stage-0 GaitGL configs, one per dataset
abnet/
  data/
    dataset.py            VideoReIDDataset, clip sampling, silhouette loading
    transforms.py         hue shift, elastic distortion, the Eq. 5 invariant
    samplers.py           P x K identity-balanced sampler (8 x 4)
    splits.py             official splits + the Appendix A probe/gallery rules
    build.py              dataset and dataloader factories
  models/
    resnet3d.py           S_phi / A_phi, vendored from 3D-ResNets-PyTorch
    gaitgl.py             T_theta, vendored from GaitGL, plus an identity head
    decoder.py            C^B and C^A, and the F_AB split
    abnet.py              the full network M
    heads.py              BNNeck identity classifier
  losses/
    classification.py     L_ce (Eq. 3), L_Ac
    triplet.py            L_tri (Eq. 4)
    distillation.py       L_KD (Eq. 1)
    distortion.py         L_Dis (Eq. 5)
  engine/
    criterion.py          Eq. 6 assembly; owns the frozen teacher
    trainer.py            training loops for both stages
    optim.py              Adam + the paper's step decay
  evaluation/
    metrics.py            CMC, mAP, TAR@FAR, the gallery filters
    evaluator.py          feature extraction and the protocol table
tools/
  extract_frames.py       videos -> JPEGs
  blur_faces.py           face-restricted setting
  extract_silhouettes.py  Mask2Former -> aligned 64x44 masks
  prepare_dataset.py      native layouts -> annotation JSON
scripts/
  train_teacher.sh        stage 0 launcher, GPU count from CUDA_VISIBLE_DEVICES
  train.sh                stage 1 launcher
  test.sh                 evaluation
train_teacher.py          stage 0
train.py                  stage 1
test.py                   evaluation
```

### How the paper maps onto the code

| paper | code |
|---|---|
| `S_φ`, ResNet3D-50 video encoder | `abnet/models/resnet3d.py` |
| `F_AB` split into two segments (§3) | `decoder.py: split_feature` |
| `C^B` actor head, `D^B_ω` (§3.1) | `decoder.py: ActorHead` |
| `f_bb`, `f_ba` via separate linear layers (§3.1) | `ActorHead.to_biometrics` / `to_appearance` |
| `C^A` activity head, `D^A_Ω` (§3.2) | `decoder.py: ActivityHead` |
| `T_θ`, GaitGL silhouette teacher (§3.1) | `abnet/models/gaitgl.py` |
| `A`, distortion network sharing `M`'s weights (§3.1) | `abnet.py: ABNetModel.forward` (second `encode` call) |
| Eq. 1, `L_KD = τ² KL(y_T ‖ y_S)` | `losses/distillation.py` |
| Eq. 2, `L_Bio = L_ce + L_tri` | `engine/criterion.py` |
| Eq. 3, `L_ce` | `losses/classification.py` |
| Eq. 4, `L_tri` with margin `m` | `losses/triplet.py` |
| Eq. 5, `L_Dis` | `losses/distortion.py` |
| Eq. 6, total objective | `engine/criterion.py: ABNetCriterion.forward` |
| activity prior at inference (§3.2) | `abnet.py: extract_features("fused")` |
| elastic distortion, α = 250 (Fig. 4) | `data/transforms.py: elastic_displacement` |
| hue shifting (§4) | `data/transforms.py: stable_hue_factor` |
| face blurring (§4) | `tools/blur_faces.py` |
| official splits + probe/gallery (App. A) | `data/splits.py: build_gallery_probe` |
| View+ / View−, same/cross activity (§4) | `evaluation/evaluator.py: PROTOCOLS` |
| rank-k, mAP, TAR @ 0.1% FAR (§4) | `evaluation/metrics.py` |

---

## Citation

```bibtex
@inproceedings{azad2024activity,
  title     = {Activity-Biometrics: Person Identification from Daily Activities},
  author    = {Azad, Shehreen and Rawat, Yogesh Singh},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {287--296},
  year      = {2024}
}
```


---

## License

Copyright © 2024 University of Central Florida. This code is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
for **non-commercial research use only**. See `LICENSE` for the full terms and
attribution.

**Third-party exception.** `abnet/models/resnet3d.py` is adapted from
[3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and
remains under the **MIT** license. It is not relicensed, and the NonCommercial /
ShareAlike conditions do not apply to it.

`abnet/models/gaitgl.py` implements the GaitGL architecture (Lin et al., ICCV
2021), written for this repository following the paper, the
[official release](https://github.com/bb12346/GaitGL) and the
[OpenGait](https://github.com/ShiqiYu/OpenGait) reference (Apache 2.0). Note
that the official release repository publishes no license file and therefore
grants no explicit permissions; nothing here should be read as a license to
that upstream code.

Mask2Former and the OpenCV face detector are fetched at runtime by the
preprocessing tools and governed by their own terms -- no weights are
distributed here. Each file credits its source in the module docstring.

### Patent notice

This repository implements technology covered, fully or partially, by two
pieces of intellectual property owned by the **University of Central Florida**:

- **US Patent 12,482,255**, *Activity-based person identification using
  biometric disentanglement* (S. Azad, Y. S. Rawat) -- issued.
- **US Patent Application 19/361,835**, *Silhouette-Guided Feature Distillation
  for Activity-Aware Person Identification* (S. Azad, Y. S. Rawat) -- pending.

**No patent rights are granted by this license.** CC BY-NC-SA 4.0 is a
copyright license only -- its Section 2(b)(2) states that "Patent and trademark
rights are not licensed under this Public License." Commercial use requires a
separate license from UCF:
[UCF Office of Technology Transfer](https://www.research.ucf.edu/tech-transfer/contact/).

**Datasets.** None are redistributed here. NTU RGB+D 120 and Charades carry
their own licences and must be obtained from their original providers, whose
terms govern their use.
