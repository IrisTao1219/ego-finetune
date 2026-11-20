"""Build datasets without loading model/GPU; print matched samples and video split."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydra import compose, initialize_config_dir
from data import build_datasets

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        cfg = compose(config_name="train", overrides=["experiment=captaincook_2xa800", *sys.argv[1:]])
    build_datasets(cfg.data, int(cfg.seed))
