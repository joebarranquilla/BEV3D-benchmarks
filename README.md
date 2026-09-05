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
