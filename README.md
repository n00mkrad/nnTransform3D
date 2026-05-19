nnTransform3D (CUDA 12.X required)

Usage: `nnTransform3D.exe input.tbc [activeVideoStart] [activeVideoEnd] [--out-mode tbc|raw] [--raw-content y|yc] [--out <path|->]`

activeVideoEnd = activeVideoStart + activeVideoWidth

Default output mode is `tbc`, which writes two files into the source directory: `input_Y.tbc` (luma) and `input_C.tbc` (chroma).

Raw mode:

- `--out-mode raw --raw-content y` writes one luma stream (`uint16` little-endian), frame-raster order `910x526`.
- `--out-mode raw --raw-content yc` writes one combined stream (`uint16` little-endian), per frame: full `Y` plane then full `C` plane.
- Raw default names are `input_Y.raw` (`y`) and `input_YC.raw` (`yc`) unless `--out` is provided.
- `--out -` is supported only in raw mode and writes binary data to `stdout`.

`--out -` is rejected in `tbc` mode because `tbc` output is dual-file.

FFmpeg decode examples (replace `30000/1001` with your actual frame rate if needed):

```bash
# Decode luma-only raw output (input_Y.raw) to a playable lossless file
ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 30000/1001 -i input_Y.raw -c:v ffv1 input_Y.mkv

# Decode combined YC raw output (input_YC.raw), then split into separate Y and C videos
ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x1052 -framerate 30000/1001 -i input_YC.raw -filter_complex "[0:v]split=2[yall][call];[yall]crop=910:526:0:0[y];[call]crop=910:526:0:526[c]" -map "[y]" -c:v ffv1 input_Y_from_YC.mkv -map "[c]" -c:v ffv1 input_C_from_YC.mkv

# Example for piped raw output from nnTransform3D (raw Y-only) directly into ffmpeg
nnTransform3D.exe input.tbc 132 896 --out-mode raw --raw-content y --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 30000/1001 -i - -c:v ffv1 piped_Y.mkv
```

Existing JSON or DB files from the input.tbc can be reused.

v2.0: Significant processing speed uplift at the cost of partial accuracy; model weights are not backward compatible with prior versions.




