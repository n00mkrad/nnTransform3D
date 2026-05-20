## nnTransform3D (CUDA 12 required)

Usage:  
`nnTransform3D.exe [--input <path>] [--av-start <num>] [--av-end <num>] [--out-mode tbc|raw_y|raw_yc|y4m] [--tbc-pipe-mode <y|c|yc_alt|yc_stack>] [--input-metadata <path>] [--input-json <path>] [--y4m-area active|full] [--y4m-first-active-frame-line <num>] [--y4m-last-active-frame-line <num>] [--out <path|->] [input.tbc]`

Options:  
`--input`: Input TBC file. Can also be passed without `--input`.  
`--av-start`: Active video area start (in pixels, horizontal)  
`--av-end`: Active video area end (in pixels, horizontal) *(Note: `av-end` - `av-start` must be positive)*  
`--out-mode`: Output mode, either `tbc`, `raw_y`, `raw_yc`, or `y4m`. Default: `tbc`.  
`--tbc-pipe-mode`: TBC stdout layout for `--out-mode tbc`: `y`, `c`, `yc_alt`, or `yc_stack`. Requires `--out -`.  
`--out`: Output path, or `-` for binary stdout. In TBC mode, `--out -` is only valid when `--tbc-pipe-mode` is set.  
`--input-metadata`: Metadata JSON path for Y4M mode.  
`--input-json`: Legacy alias for `--input-metadata`.  
`--y4m-area`: Y4M output area, `active` or `full`. Default: `active`.  
`--y4m-first-active-frame-line`: First active frame line for Y4M active-area mode (default `40`).  
`--y4m-last-active-frame-line`: Last active frame line for Y4M active-area mode (default `525`, exclusive).  

### Active Video Area

Can be read from your TBC.json or DB file (or GUIs like ld-analyse).

Hint: activeVideoEnd = activeVideoStart + activeVideoWidth

### Y4M Output Mode

- `--out-mode y4m` writes YUV4MPEG2 `YUV444P16` limited-range frames.
- Uses metadata JSON from `--input-metadata` / `--input-json`, or auto-falls back to `<input>.json`.
- Video is merged from separated luma and chroma using minimal `mono`/`ntsc1d` decoders. More advanced comb filters are not needed as Y/C is already cleanly separated.
- `--y4m-area active` crops to active picture (`videoParameters.activeVideoStart/End` and configurable frame-line range).
- `--y4m-area full` outputs full metadata frame geometry (`fieldWidth x ((fieldHeight * 2) - 1)`).
- Explicit `--av-start/--av-end` can override metadata horizontal bounds in `active` area mode.
- `--out -` is supported for piping Y4M to stdout.

Examples:

```bash
# Default Y4M file output (auto metadata from input.tbc.json)
nnTransform3D --input input.tbc --out-mode y4m

# Explicit metadata file and active-area output to stdout
nnTransform3D --input input.tbc --out-mode y4m --input-metadata tbc-example.json --y4m-area active --out - > output.y4m

# Full-frame Y4M output
nnTransform3D --input input.tbc --out-mode y4m --input-json tbc-example.json --y4m-area full --out output_full.y4m
```

### Default TBC File Output

Default output mode is `tbc`, which writes two files into the source directory: `input_Y.tbc` (luma) and `input_C.tbc` (chroma).

### TBC Stdout Pipe Modes

All TBC pipe modes emit headerless `uint16` little-endian samples and preserve current field-sequential TBC frame chunk packing.

- `--tbc-pipe-mode y`: Emit only luma TBC frame chunks.
- `--tbc-pipe-mode c`: Emit only chroma TBC frame chunks.
- `--tbc-pipe-mode yc_alt`: Emit luma chunk then chroma chunk for each source frame. Interpret as `910x526` with doubled frame cadence. (6000/1001 FPS with alternating Y/C)
- `--tbc-pipe-mode yc_stack`: Emit the same byte order as `yc_alt`, but interpret as vertically stacked `910x1052` frames (Y top, C bottom). This is intentionally out-of-spec TBC geometry.

`yc_alt` and `yc_stack` produce identical bytes; only downstream interpretation differs.

Examples:

```bash
# Y-only TBC stream to file
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode tbc --tbc-pipe-mode y --out - > input_Y.tbc

# C-only TBC stream to file
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode tbc --tbc-pipe-mode c --out - > input_C.tbc

# YC alternating mode interpreted as 910x526 at 2x frame cadence
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode tbc --tbc-pipe-mode yc_alt --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 60000/1001 -i - -c:v ffv1 input_YC_alt.mkv

# YC stacked interpretation (910x1052), split back into Y and C
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode tbc --tbc-pipe-mode yc_stack --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x1052 -framerate 30000/1001 -i - -filter_complex "[0:v]split=2[yall][call];[yall]crop=910:526:0:0[y];[call]crop=910:526:0:526[c]" -map "[y]" -c:v ffv1 input_Y_from_stack.mkv -map "[c]" -c:v ffv1 input_C_from_stack.mkv
```

### Raw Video Output Mode

- `--out-mode raw_y` writes one luma stream (`uint16` little-endian), frame-raster order `910x526`.
- `--out-mode raw_yc` writes luma and chroma into one stream (`uint16` LE) by stacking them vertically, resulting in `910x1052` frames (luma on top, chroma on bottom).
- Raw default names are `input_Y.raw` (`raw_y`) and `input_YC.raw` (`raw_yc`) unless `--out` is provided.
- `--out -` can be used in raw mode to pipe the data directly into another process (e.g. ffmpeg) without writing to disk.

FFmpeg and mpv decode examples (replace `30000/1001` with your actual frame rate if needed):

```bash
# Decode raw luma output (input_Y.raw) to a lossless FFV1 MKV
ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x526 -framerate 30000/1001 -i input_Y.raw -c:v ffv1 input_Y.mkv

# Pipe to ffmpeg and decode combined YC raw output while splitting it into separate Y and C videos and encode as lossless FFV1 MKV
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode raw_yc --out - | ffmpeg -f rawvideo -pixel_format gray16le -video_size 910x1052 -framerate 30000/1001 -i - -filter_complex "[0:v]split=2[yall][call];[yall]crop=910:526:0:0[y];[call]crop=910:526:0:526[c]" -map "[y]" -c:v ffv1 input_Y_from_YC.mkv -map "[c]" -c:v ffv1 input_C_from_YC.mkv

# Pipe to mpv to preview luma output in real time
nnTransform3D --input input.tbc --av-start 132 --av-end 896 --out-mode raw_y --out - | mpv --demuxer=rawvideo --demuxer-rawvideo-mp-format=gray16le --demuxer-rawvideo-w=910 --demuxer-rawvideo-h=526 --demuxer-rawvideo-fps=30000/1001 -
```

