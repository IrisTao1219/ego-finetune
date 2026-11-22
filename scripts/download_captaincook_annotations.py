"""Download CaptainCook4D instruction annotations once, without loading a model."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from datasets.common import _load_hf_dataset_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True,
                        help='Same directory as data_paths.captaincook4d_root')
    parser.add_argument('--include-validation', action='store_true',
                        help='Also download main_val.json; not needed by the two-A800 config')
    args = parser.parse_args()
    root = Path(args.data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    splits = [('train', 'train')]
    if args.include_validation:
        splits.append(('validation', 'val'))
    for remote_split, local_split in splits:
        destination = root / f'main_{local_split}.json'
        if destination.exists():
            with destination.open() as handle:
                rows = json.load(handle)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f'Invalid existing annotation file: {destination}')
            print(f'Already present: {destination} ({len(rows)} rows)', flush=True)
            continue
        rows = _load_hf_dataset_rows(
            'qualcomm/qualcomm-interactive-cooking-dataset', 'main', [remote_split])
        required = {'video_id', 'output_texts', 'output_types', 'output_timestamps'}
        if not rows or any(not required.issubset(row) for row in rows):
            raise ValueError(f'Invalid or empty {remote_split} annotations; not saved')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=root, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(rows, handle, ensure_ascii=False)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'Saved {destination} ({len(rows)} rows)', flush=True)


if __name__ == '__main__':
    main()
