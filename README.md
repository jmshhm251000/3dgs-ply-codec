# 3DGS PLY Codec

A rate–distortion-driven compressor for **3D Gaussian Splatting** scenes, written
in C++/CUDA. Goal: push a trained `.ply` down-and-left on the R-D plane
(x = file size, y = rendered PSNR), beating the standard PNG-compression baseline.

## Pipeline

```
.ply (N gaussians x ~59 floats)
  [0] Ordering      Morton / PLAS       group similar gaussians
  [1] Transform     log / delta / PCA   re-express into a codeable shape
  [2] Quantization  SQ / VQ             floats -> integer symbols   (loss)
  [3] Entropy code  histogram+arith /   symbols -> bits             (shrink)
                    learned context
```

The codec is a **per-component routing** of this framework: each attribute
(position, scale, rotation, opacity, SH-DC, SH-rest) flows through a chosen subset
of stages, decided by measurement rather than hardcoding.

## Status

- **M0 — rate (done).** `plycodec` streams a `.ply`, min-max scalar-quantizes to
  `b` bits, and reports the per-component Shannon-entropy floor `H_c(b)` — the
  rate leg of the R-D curve. See `docs/DESIGN_JOURNAL.md`.
- **M1 — distortion (in progress).** A CUDA forward rasterizer measures
  `PSNR(render(original), render(quantized))` per component/bit-depth = `D_c(b)`,
  the distortion leg. Build spec: `docs/CUDA_RASTERIZER.md`.
  `tools/reference_gsplat.png` is the correctness oracle.

## Build & run (codec)

```bash
cmake -S . -B build
cmake --build build
build/plycodec <file.ply> <chunk_size> <bits>      # e.g. bonsai.ply 65536 8
```

Requires a C++17 compiler with AVX2 (the quantizer SIMD path). CUDA is added for
the renderer (see the rasterizer spec).

## Layout

```
include/           quantizer, plyparser, rd_profile headers
src/               codec implementation + main
docs/
  DESIGN_JOURNAL.md   design decisions, research (primary-verified), M0 findings
  CUDA_RASTERIZER.md  the renderer build spec (M1)
tools/
  reference_gsplat.png  render oracle for verifying the CUDA rasterizer
CMakeLists.txt
```

## Notes

Findings are grounded in the 3DGS-compression literature (c3dgs, HAC, SOG,
MesonGS — primary sources verified), and validated on `bonsai.ply` (a degree-0
scene: all SH-rest is zero, and `f_dc`/`opacity` are pre-quantized to 8-bit — see
the design journal for what that implies).
