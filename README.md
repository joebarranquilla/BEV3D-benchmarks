# BEV 3D Detection Benchmark (nuScenes-mini)

Simple benchmark harness to compare BEV-style 3D detection models on nuScenes-mini:
- BEVDepth
- LSS (vanilla baseline)
- BEVFormer

This version keeps things minimal:
- One shared config style (base + model override)
- One run command interface for all models
- One result format (JSON) and one leaderboard table (CSV)
- Quick visualizations (mAP and NDS bar charts)

## 1) Setup

```bash
cd bev3d-nuscenes-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 2) Dataset (nuScenes-mini)

```bash
bash scripts/download_nuscenes_mini.sh
```

Then manually download `v1.0-mini.tgz` from https://www.nuscenes.org/download and extract into `data/nuscenes`.

Expected structure:

```text
data/
  nuscenes/
    v1.0-mini/
    samples/
    sweeps/
    maps/
```

## 3) Clone model repositories

```bash
cp configs/model_repo_urls.example.env configs/model_repo_urls.env
# edit configs/model_repo_urls.env with git URLs
bash scripts/clone_model_repos.sh
```

Manual alternative:

```bash
mkdir -p third_party
git clone <BEVDepth_repo_url> third_party/BEVDepth
git clone <LSS_repo_url> third_party/LSS
git clone <BEVFormer_repo_url> third_party/BEVFormer
```

These repo folders are referenced directly by:
- `configs/models/bevdepth.yaml`
- `configs/models/lss.yaml`
- `configs/models/bevformer.yaml`

## 4) Download checkpoints

### Option A: Fill env file + helper script

```bash
cp configs/checkpoint_urls.example.env configs/checkpoint_urls.env
# edit configs/checkpoint_urls.env and paste real URLs
bash scripts/download_checkpoints.sh
```

### Option B: Manual commands

```bash
mkdir -p checkpoints
wget -O checkpoints/bevdepth_r50.pth "<BEVDEPTH_CHECKPOINT_URL>"
wget -O checkpoints/lss_r50.pth "<LSS_CHECKPOINT_URL>"
wget -O checkpoints/bevformer_base.pth "<BEVFORMER_CHECKPOINT_URL>"
```

## 5) Dry-run and evaluate

Dry-run command preview (no model execution):

```bash
bash scripts/run_baseline.sh bevdepth
bash scripts/run_baseline.sh lss
bash scripts/run_baseline.sh bevformer
```

Real execution (runs underlying test command):

```bash
EXECUTE=1 bash scripts/run_baseline.sh bevdepth
EXECUTE=1 bash scripts/run_baseline.sh lss
EXECUTE=1 bash scripts/run_baseline.sh bevformer
```

Each run writes a standardized JSON file into `results/`.

## 5b) BEVDepth environment (required before EXECUTE=1)

BEVDepth pins `torch==1.9.0`/`torchvision==0.10.0` and needs Python <= 3.9, which is
incompatible with this project's `.venv` (Python 3.13, no CUDA toolchain). Building
BEVDepth's custom CUDA ops also needs `nvcc` matching CUDA 11.1 and a host compiler
`gcc <= 10` (modern system `gcc` is too new). The cleanest fix is a dedicated conda
env that bundles its own CUDA toolkit and compiler, so no system/admin CUDA install
is required:

```bash
conda create -y -n bevdepth -c pytorch -c nvidia -c conda-forge \
  python=3.8 \
  pytorch=1.9.0=py3.8_cuda11.1_cudnn8.0.5_0 \
  torchvision=0.10.0 \
  cudatoolkit=11.1 \
  cudatoolkit-dev=11.1.1 \
  gxx_linux-64=9 \
  gcc_linux-64=9

conda activate bevdepth

# conda-forge's cudatoolkit-dev pulls in a too-new mkl that breaks torch import
# (undefined symbol: iJIT_NotifyEvent) - pin it back down:
conda install -c pytorch -c defaults -y "mkl=2021.4.0" "blas=1.0=mkl"

# pip>=24.1 rejects pytorch-lightning==1.6.0's malformed metadata:
pip install "pip==24.0"

# Install deps with --no-deps where noted so nothing pulls in a newer torch:
pip install numba numpy nuscenes-devkit opencv-python-headless pandas scikit-image \
  scipy "setuptools==59.5.0" tensorboardX
pip install --no-deps "pytorch-lightning==1.6.0" "torchmetrics==0.7.3" "pyDeprecate==0.3.2"

# Build BEVDepth's CUDA ops (voxel pooling) with the conda-provided nvcc/gcc:
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
pip install -e third_party/BEVDepth

# BEVDepth's backbones need mmcv/mmdet/mmdet3d (OpenMMLab stack). Versions below
# are the overlapping range that satisfies mmdet3d v1.0.0rc4, mmdet, and mmseg:
pip install "mmcv-full==1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install "mmdet==2.24.1" "mmsegmentation==0.29.1"
pip install lyft_dataset_sdk "networkx==2.8.8" "numba==0.58.1" "llvmlite==0.41.1" plyfile \
  "trimesh>=2.35.39,<2.35.40"

git clone --depth 1 --branch v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git third_party/mmdetection3d
cd third_party/mmdetection3d && pip install --no-deps -e . && cd -
```

Verify it works:

```bash
conda run -n bevdepth python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n bevdepth python -c "from bevdepth.exps.base_cli import run_cli"
```

`configs/models/bevdepth.yaml` sets `python_executable` to this conda env's
`bin/python` so `scripts/run_baseline.sh` launches the BEVDepth subprocess with the
right interpreter regardless of which env the harness itself runs under.

## 6) Build leaderboard + visualizations

```bash
bevbench-report --results-dir results --out-csv results/leaderboard.csv
bevbench-viz --leaderboard results/leaderboard.csv --out-dir results/plots
```

Expected outputs:
- `results/leaderboard.csv`
- `results/plots/map.png`
- `results/plots/nds.png`

## 7) Important notes

- `src/bevbench/runners.py` runs a simple default eval command (`python tools/test.py <config> <ckpt> --eval bbox`).
- Different repos may require slightly different flags; adjust `_build_eval_command()` as needed.
- To include metrics in the leaderboard, place each model's metric summary JSON at the path configured in `evaluation.metrics_file` inside each model YAML.

## 8) TruckScenes support (man-truckscenes)

You can run the same harness on:

```text
data/truckscenes/man-truckscenes
```

### 8.1 Prepare TruckScenes metadata and info PKLs

TruckScenes metadata is close to nuScenes, but not directly drop-in for the
existing BEVDepth/BEVFormer eval scripts. Generate compatibility files with:

```bash
/usr/prakt/s0012/miniconda3/envs/bevformer/bin/python scripts/prepare_truckscenes.py \
  --dataset-root data/truckscenes/man-truckscenes \
  --source-version v1.2-mini \
  --target-version v1.0-mini
```

This writes:

- `data/truckscenes/man-truckscenes/v1.0-mini/*.json` (normalized metadata)
- `data/truckscenes/man-truckscenes/nuscenes_infos_train.pkl`
- `data/truckscenes/man-truckscenes/nuscenes_infos_val.pkl`
- `data/truckscenes/man-truckscenes/nuscenes_infos_temporal_train.pkl`
- `data/truckscenes/man-truckscenes/nuscenes_infos_temporal_val.pkl`

### 8.2 Run baseline on TruckScenes

Dry run:

```bash
bash scripts/run_baseline.sh bevdepth truckscenes
bash scripts/run_baseline.sh bevformer truckscenes
```

Execute:

```bash
EXECUTE=1 bash scripts/run_baseline.sh bevdepth truckscenes
EXECUTE=1 bash scripts/run_baseline.sh bevformer truckscenes
```

TruckScenes model configs are in:

- `configs/models/truckscenes/bevdepth.yaml`
- `configs/models/truckscenes/bevformer.yaml`

Custom external config/entry files used by those configs:

- `configs/external/bevdepth_truckscenes_eval.py`
- `configs/external/bevformer_tiny_truckscenes.py`

### 8.3 Assumptions behind the TruckScenes run scripts

The current scripts make these assumptions to run BEVDepth/BEVFormer unchanged:

1. **NuScenes API compatibility is emulated for TruckScenes**
   - `scripts/prepare_truckscenes.py` materializes `v1.0-mini` metadata under
     `data/truckscenes/man-truckscenes/v1.0-mini`.
   - Missing nuScenes tables (`log.json`, `map.json`) are synthesized.

2. **Evaluation split behavior is forced through scene-name remapping**
   - Scene names are remapped to official nuScenes mini split names so
     `mini_val` filtering in the nuScenes evaluator includes all TruckScenes samples.

3. **Sensor naming is normalized to nuScenes conventions**
   - A reference TruckScenes lidar channel is aliased to `LIDAR_TOP`.
   - Four TruckScenes cameras are aliased into six canonical nuScenes camera slots:
     `CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`,
     `CAM_BACK_LEFT`, `CAM_BACK`, `CAM_BACK_RIGHT`.
   - This keeps BEVFormer/BEVDepth camera-count assumptions and pretrained heads intact.

4. **Annotation attributes are restricted to nuScenes detection attributes**
   - Non-nuScenes attribute tokens are dropped from `sample_annotation.json`
     in the normalized metadata to avoid assertion failures in nuScenes eval.

5. **Two different info PKL schemas are required**
   - BEVDepth consumes `nuscenes_infos_{train,val}.pkl`.
   - BEVFormer consumes `nuscenes_infos_temporal_{train,val}.pkl`.
   - Both are generated by `scripts/prepare_truckscenes.py` and must exist before
     `EXECUTE=1` runs.

6. **Model checkpoints are still nuScenes checkpoints**
   - Reported TruckScenes metrics are zero-shot transfer numbers and can be very low.
   - Good TruckScenes accuracy requires fine-tuning on TruckScenes labels.

7. **Environment assumptions**
   - BEVDepth run uses the dedicated conda env configured in
     `configs/models/truckscenes/bevdepth.yaml` (`python_executable`).
   - BEVFormer run uses the dedicated conda env configured in
     `configs/models/truckscenes/bevformer.yaml`.
   - `scripts/run_baseline.sh` only switches configs/orchestration; it does not
     auto-create environments or install dependencies.

8. **Execution assumptions in the harness**
   - `bash scripts/run_baseline.sh <model> truckscenes` is a dry run (command planning).
   - `EXECUTE=1 ...` runs the external model repository command and writes logs to
     `results/logs/<model>_last.log`.
