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
    datasets = build_datasets(cfg.data, int(cfg.seed))
    train = datasets["train"]
    import json
    if hasattr(train, "sample_location"):
        previous = set()
        for epoch in range(2):
            train.set_epoch(epoch)
            locations = [train.sample_location(i) for i in range(len(train))]
            unique = set(locations)
            report = {"epoch_index": epoch, "draws": len(locations),
                      "unique_samples": len(unique),
                      "repeated_draws": len(locations) - len(unique),
                      "new_vs_previous_epoch": len(unique - previous),
                      "components": []}
            for idx, component in enumerate(train.datasets):
                report["components"].append({
                    "component_index": idx, "raw_samples": len(component),
                    "draws": sum(c == idx for c, _ in locations),
                    "unique_samples": sum(c == idx for c, _ in unique)})
            print("SAMPLING_REPORT " + json.dumps(report), flush=True)
            previous = unique
        train.set_epoch(0)
