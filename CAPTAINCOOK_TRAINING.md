# CaptainCook4D subset: two A800 80GB GPUs

Run from this repository root on the CUDA server. Use the repository Docker/dependency environment; no GPU run has been verified locally.

Videos: `../data/captain_cook_4d/gopro/resolution_360p/<id>_360p.mp4`.
Extract whole videos at 2 fps into `../data/captain_cook_4d/gopro/resolution_360p_2fps_frames/<id>_360p/frame_%06d.jpg` using ffmpeg. Keep the original time origin; do not extract trimmed clips. Remove/rebuild stale `../data/video_to_frames.pkl` after adding frames.

```bash
bash scripts/extract_captaincook_frames.sh
python scripts/check_captaincook_data.py
CUDA_VISIBLE_DEVICES=0,1 python scripts/run.py experiment=captaincook_2xa800 trainer.max_steps=100 trainer.max_epochs=1
CUDA_VISIBLE_DEVICES=0,1 python scripts/run.py experiment=captaincook_2xa800
```

The data check downloads annotations on first use and builds the frame cache. Read its `VIDEO_SPLIT` output: each split lists video IDs, component sample counts (instructions, counterfactual), and length. Ensure both components have matched samples. Split assignment is deterministic for the same matched videos and seed; adding videos changes it, so freeze data during an experiment. Save stdout with the experiment artifacts.

The new configuration uses only CaptainCook4D frames, including CaptainCook4D rows from Ego-CoMist. Other source paths are placeholders required by the existing loader; no frames from those sources are selected. Instruction annotations use the upstream train split only. The new video split is a local 80/10/10 split, not the official benchmark split. Missing videos are skipped by the loaders. Sampling weights are 1:2 (instructions:counterfactual), a local subset choice rather than the full paper mixture.

Training uses BF16, LoRA, action head, gradient checkpointing, existing ZeRO-3/CPU offload, micro-batch 1 per GPU, accumulation 16 (effective batch 32), and at most 128 frames at 2 fps / 224 pixels. The reduced frame limit changes the visible context relative to the default 200-frame configuration. First measure memory and throughput; two 80GB GPUs do not guarantee a particular batch fits.

There are 3200 training draws per epoch and a 1000-update target over 10 epochs. Cosine T_max follows trainer.max_steps. Validation/test enumerate held-out raw samples without weighting or temporal jitter; video IDs are shared across component split assignments, preventing cross-component video leakage. Original default experiment behavior is unchanged.

To use a local model, append `model.tokenizer.pretrained_model_name_or_path=/absolute/path/to/Qwen3.5-2B`. To change data location, append `data_paths.captaincook4d_root=/absolute/path/to/datasets` to both commands. Download annotations/model before enabling offline mode. Do not train on Ego-MC-Bench evaluation videos.

## Environment aligned with ego-mc-bench-test

`requirements-base.txt` copies the tested inference project baseline: torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers >=5.5.0,<6, accelerate >=1.0, opencv-python-headless >=4.10, Pillow >=10. The training requirements include this file and add DeepSpeed, PEFT, Hydra and the streaming kernel package. They no longer pin Transformers 5.2.0 or install the GUI OpenCV distribution. This is a shared version baseline, not proof that training/backpropagation or compiled extensions have been validated.

In the existing Linux CUDA server environment, run from this repository root:

```bash
conda create -n captaincook-train python=3.10 -y
conda activate ego-train

python -m pip install --dry-run -r docker/requirements-pip.txt
python -m pip install -r docker/requirements-pip.txt
python -m pip check
python scripts/check_kernels.py


# 1. 检查内核
python scripts/check_kernels.py

# 2. 还没抽帧的话，先执行
bash scripts/extract_captaincook_frames.sh

# 3. 检查匹配样本和视频划分
python scripts/check_captaincook_data.py
```

Do not run these installation commands on the local Mac. The dry run lets you review dependency changes before installing. Existing packages satisfying the bounds are reused (no --upgrade). If both opencv-python and opencv-python-headless are already installed, resolve that shared cv2 namespace conflict in the server environment before training.

The Dockerfile retains the original CUDA 12.8 development base for nvcc, but explicitly installs the shared torch pin before compiling causal-conv1d. An existing server environment also needs a compatible causal-conv1d build for its installed torch; if the kernel check fails, use the repository Docker build or rebuild that extension against the server torch. Inference success alone does not validate DeepSpeed, PEFT, FLA or causal-conv1d training compatibility. No local torch version is used to choose these requirements.

## Prefetch instruction annotations before training

The CaptainCook-only configuration now reads instruction annotations locally. First run:

```bash
python scripts/download_captaincook_annotations.py --data-root ../data
python scripts/check_captaincook_data.py data_paths.captaincook4d_root=../data
```

This writes `../data/main_train.json` using the same HF repo/config/train split as before. The loader honors `include_validation_in_train: false` for local annotations too, so no `main_val.json` is required and the upstream validation split is not silently added. Other experiments retaining the default true value still require both files; pass `--include-validation` to prefetch both. Completed local files are reused, downloads are validated and published atomically. Only this preparation step requires network access for missing annotations; training reads the instruction JSON and the configured local Ego-CoMist JSON. Do not substitute Ego-CoMist's annotations.json for main_train.json.

## Checkpoints and interrupted training

Artifacts are separate from stdout logs and Git changes. The run prints its
absolute output directory and checkpoint save start/completion. DeepSpeed writes
`*model_states.pt` and `*optim_states.pt` shards under `checkpoints/`; a standalone
HF model file is not generated (`export_fp32_weights` defaults to false).
CaptainCook saves every 25 optimizer updates and at `max_steps` before validation,
as well as after each epoch validation. Step checkpoints are not pruned by
`save_top_k`, so adjust the interval for available disk space.

Resume on the CUDA server using the same model, data, GPU count, batch size and
accumulation settings as the interrupted run. Pass a checkpoint **directory**, not
one individual `.pt` shard:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run.py experiment=captaincook_2xa800 \
  data_paths.captaincook4d_root=/home/taosha/data \
  model.tokenizer.pretrained_model_name_or_path=/home/taosha/datasets/Hf_model/Qwen3.5-2B \
  ckpt_path=/home/taosha/ego-finetune/outputs/logs/train/captaincook_subset/runs/2026-11-22_14-55-28/checkpoints
```

A checkpoint root loads its `latest` tag; append `/epoch-8` or `/step-25` to select
one explicitly. Full DeepSpeed checkpoints restore model, optimizer, scheduler,
epoch and global update count. A consolidated fp32 `.pt` file only initializes
model weights; it does not resume optimizer or training progress.

New step checkpoints also store the batch cursor and accumulated training metrics;
resume skips the already processed batch positions in that epoch. Legacy step
checkpoints without a cursor restart that epoch (with a warning). Epoch checkpoints
start at the next epoch. Random generator/dataset/worker states are not fully
restored, so this is practical continuation, not an identical stochastic trajectory.
Updates after the most recent checkpoint must be rerun; interrupting the process
does not itself trigger an emergency save.

`auto_resume=true` now loads `checkpoints/latest` in the **current output directory**
when no explicit `ckpt_path` is given. Since Hydra creates a new directory each run,
set `hydra.run.dir=/absolute/path/to/previous/run` to use this option. Explicit
`ckpt_path` with a fresh output directory keeps the previous run's artifacts intact.
`max_steps` and `max_epochs` are total targets, not additional training amounts;
a run already at its target performs no further training. To extend completed
training, increase both limits as needed. A restored scheduler retains its saved
schedule; changing `max_steps` does not automatically extend that schedule.

These paths and save/resume branches have been checked locally with mock engines;
actual save/load on two CUDA GPUs still requires server verification.

## Best model selection

`save_top_k` now retains the K epoch checkpoints with the lowest validation total
loss, plus the latest epoch for resuming (up to K+1 epoch folders). `-1` keeps all;
`0` keeps only the latest epoch. Step checkpoints remain separate and unpruned.
`checkpoints/best` contains the selected epoch tag; `checkpoints/latest` remains
the most recent checkpoint for continuation. `epoch_scores.json` persists scores
across restarts in the same output directory. Unindexed legacy epoch folders are
left untouched; checkpoints deleted by previous code cannot be recovered.

For evaluation/export, read the `best` marker and pass that complete directory:

```bash
CHECKPOINT_ROOT=/absolute/path/to/run/checkpoints
BEST_TAG=$(cat "$CHECKPOINT_ROOT/best")
python scripts/export_zero_checkpoint.py "$CHECKPOINT_ROOT/$BEST_TAG" /absolute/path/to/export
```

Final test metrics and automatic fp32 export still refer to the final trained
model, not automatically to the best checkpoint. Export/evaluate `best` explicitly
when reporting the selected model's results.

## Audit loading and sampling before another run

Run on the CUDA server (no training model/GPU is loaded):

```bash
set -o pipefail
python scripts/check_captaincook_data.py \
  data_paths.captaincook4d_root=/home/taosha/data \
  2>&1 | tee captaincook-data-audit.log
```

`DATA_LOAD_REPORT` shows annotation rows, source/config exclusions, successful
samples/videos and skip counts. For instructions, candidate instructions—not
annotation rows—are the sample denominator (one row can contain multiple steps).
For Ego-CoMist, source totals and matched source totals are reported. Missing
video frames and empty time windows are counted separately; missing instruction/
response content is also reported. These are loader decisions, not proof that
cached frame files are all readable or that timestamps/labels are correct.

`VIDEO_SPLIT` shows video-disjoint splits and raw component counts.
`SAMPLING_REPORT` shows draws, unique samples, repeated draws and per-component
coverage for epoch indices 0 and 1. This audit resolves sample indices without
loading image pixels. Source filtering is intentional in the CaptainCook-only
configuration; it does not represent the full paper mixture.

Training now changes the weighted sample selection each epoch, deterministically
from seed/epoch/index. Component sample weights are applied once. Sampling remains
with replacement; 3200 draws need not cover every raw sample. Worker processes
are recreated each epoch for epoch-aware mixtures so they receive the new epoch.
Validation/test still enumerate held-out samples without this resampling.
The sampling algorithm has changed, so old runs cannot be reproduced exactly
with the new code, including when continuing an old checkpoint. Training length,
learning rate and mixture proportions have not been changed by this repair.
