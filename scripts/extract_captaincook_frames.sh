#!/usr/bin/env bash
# Run from the repository root; completed videos are resumable.
set -euo pipefail
src="${1:-../datasets/captain_cook_4d/gopro/resolution_360p}"
dst="${2:-../datasets/captain_cook_4d/gopro/resolution_360p_2fps_frames}"
command -v ffmpeg >/dev/null
shopt -s nullglob
videos=("$src"/*_360p.mp4)
if (( ${#videos[@]} == 0 )); then
    echo "No *_360p.mp4 videos found in $src" >&2
    exit 1
fi
for video in "${videos[@]}"; do
    name="$(basename "$video" .mp4)"
    out="$dst/$name"
    [[ -f "$out/.complete" ]] && continue
    # Reject partial output rather than silently mixing old and new frames.
    if [[ -d "$out" ]] && compgen -G "$out/*.jpg" >/dev/null; then
        echo "Incomplete frame directory: $out. Move it aside before retrying." >&2
        exit 1
    fi
    mkdir -p "$out"
    ffmpeg -nostdin -n -i "$video" -vf 'scale=-2:360,fps=2' -q:v 2 "$out/frame_%06d.jpg"
    touch "$out/.complete"
done
echo 'Extraction complete. Rebuild stale video_to_frames.pkl before the data check.'
