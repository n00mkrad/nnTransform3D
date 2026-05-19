## nnTransform3D (CUDA 12 required)

Usage:  
`nnTransform3D.exe [--input <path>] [--av-start <num>] [--av-end <num>] [--out-mode tbc|raw_y|raw_yc] [--out <path|->] [input.tbc]`

Options:  
`--input`: Input TBC file. Can also be passed without `--input`.  
`--av-start`: Active video area start (in pixels, horizontal)  
`--av-end`: Active video area end (in pixels, horizontal) *(Note: `av-end` - `av-start` must be positive)*  
`--out-mode`: Output mode, either `tbc`, `raw_y`, or `raw_yc`. Default: `tbc`.  
`--out`: Output path or `-` for binary stdout (only in raw mode) for piping. Default is to write files in the source directory.  

### Active Video Area

Can be read from your TBC.json or DB file (or GUIs like ld-analyse).

Hint: activeVideoEnd = activeVideoStart + activeVideoWidth

Default output mode is `tbc`, which writes two files into the source directory: `input_Y.tbc` (luma) and `input_C.tbc` (chroma).

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

