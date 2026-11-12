#!/usr/bin/env bash
# Re-cut every page clip and poster from the full-resolution supplementary video.
#
#   tools/cut_clips.sh /path/to/supplementary_master.mp4 [height]
#
# height defaults to 720 (1280x720). The timestamps below were taken from the
# 3:42 preview cut (final_v2_x264_preview.mp4). If your master is edited
# differently, adjust START/END in the table. Requires ffmpeg.
set -euo pipefail
SRC="$1"; H="${2:-720}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/assets"
mkdir -p "$OUT/video" "$OUT/posters"
# square the pixels first (the preview cut is anamorphic 720x480, SAR 32:27), then scale to the target height
VF="scale=trunc(iw*sar/2)*2:ih,setsar=1,scale=-2:${H}:flags=lanczos,setsar=1"

#  file                 start  end    poster time (seconds in the source video)
while read -r NAME START END POSTER; do
  [ -z "$NAME" ] && continue
  ffmpeg -nostdin -v error -y -ss "$START" -to "$END" -i "$SRC" -an -vf "$VF" \
    -c:v libx264 -preset slow -crf 21 -profile:v high -pix_fmt yuv420p -g 48 -movflags +faststart \
    "$OUT/video/$NAME.mp4"
  ffmpeg -nostdin -v error -y -ss "$POSTER" -i "$SRC" -frames:v 1 -vf "$VF" -q:v 3 "$OUT/posters/$NAME.jpg"
  printf '%-20s %s\n' "$NAME" "$(du -h "$OUT/video/$NAME.mp4" | cut -f1)"
done <<'LIST'
teaser                3.000000 15.291667 6.000000
collect_operator      15.291667 23.958333 18.875000
collect_headcam       27.000000 32.500000 27.875000
design_cad            33.833333 41.375000 35.375000
lift_box              42.708333 49.833333 44.375000
push_chair            49.833333 62.208333 53.875000
push_chair_spatial    62.208333 74.791667 65.875000
push_chair_env        74.791667 84.250000 77.875000
zip_tie               84.250000 101.250000 89.875000
lego                  101.250000 126.916667 113.875000
lego_strategy         126.916667 146.958333 134.875000
box_conveyor          146.958333 186.875000 169.875000
efficiency            186.916667 219.875000 201.875000
LIST

# Full video with audio (or delete it and use a YouTube embed, see README).
ffmpeg -nostdin -v error -y -i "$SRC" -vf "$VF" -c:v libx264 -preset slow -crf 23 -profile:v high \
  -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k "$OUT/video/supplementary.mp4"
ffmpeg -nostdin -v error -y -ss 6.0 -i "$SRC" -frames:v 1 -vf "$VF" -q:v 3 "$OUT/posters/supplementary.jpg"
echo "supplementary        $(du -h "$OUT/video/supplementary.mp4" | cut -f1)"
