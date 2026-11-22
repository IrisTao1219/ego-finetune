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
python -m pip install --dry-run -r docker/requirements-pip.txt
python -m pip install -r docker/requirements-pip.txt
python -m pip check
python scripts/check_kernels.py
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
