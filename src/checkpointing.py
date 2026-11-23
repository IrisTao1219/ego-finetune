"""Validation-ranked retention for complete DeepSpeed epoch checkpoints."""

import json
import math
import re
import shutil
from pathlib import Path


def retain_epoch_checkpoints(root: Path, tag: str, val_loss: float, top_k: int) -> None:
    """Keep the best K plus latest epoch; run on rank zero after all ranks save.

    The index persists across restarts. Unindexed legacy checkpoints are left alone.
    A negative top_k keeps all checkpoints; zero keeps only the latest epoch.
    """
    if not re.fullmatch(r"epoch-\d+", tag):
        raise ValueError(f"Invalid epoch checkpoint tag: {tag}")
    index_path = root / "epoch_scores.json"
    scores = json.loads(index_path.read_text()) if index_path.exists() else {}
    scores = {
        name: score for name, score in scores.items()
        if re.fullmatch(r"epoch-\d+", name) and (root / name).is_dir()
    }
    scores[tag] = float(val_loss) if math.isfinite(val_loss) else None
    ranked = sorted(
        (name for name in scores if scores[name] is not None),
        key=lambda name: (scores[name], int(name.split("-")[1])),
    )
    # Write the score index before deleting anything, so restart can finish pruning.
    temporary = root / "epoch_scores.json.tmp"
    temporary.write_text(json.dumps(scores, indent=2) + "\n")
    temporary.replace(index_path)
    if ranked:
        temporary = root / "best.tmp"
        temporary.write_text(ranked[0] + "\n")
        temporary.replace(root / "best")
    keep = set(scores if top_k < 0 else ranked[:top_k]) | {tag}
    # Protect the engine's latest marker too, including a newer step checkpoint.
    if (root / "latest").is_file():
        keep.add((root / "latest").read_text().strip())
    for name in scores.keys() - keep:
        shutil.rmtree(root / name)
    if ranked and ranked[0] not in keep:
        (root / "best").unlink(missing_ok=True)
