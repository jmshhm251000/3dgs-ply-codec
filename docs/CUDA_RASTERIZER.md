# CUDA Forward Rasterizer — Build Spec

> **Purpose.** Build a CUDA forward Gaussian-splat rasterizer in C++ to serve as
> the **distortion meter** for the codec. It renders a 3DGS scene to an image so
> we can measure PSNR between `render(original)` and `render(quantized)` — the
> distortion leg of the rate–distortion curve. Pure C++/CUDA, no Python.
>
> This is a solo continuation spec: everything you need to build it on the laptop
> without the chat context. Work through the stages **in order**; each has an
> acceptance check.

---

## 0. Context & what is already settled

- The codec (C++) measures **rate** (`plycodec.exe`, `src/rd_profile.cpp`). This
  renderer measures **distortion**. Together → the R-D curve.
- **Ablation loop (the goal):** for each component × bit depth, quantize only that
  component (min-max SQ, same as `Quantizer`), render, `PSNR(base, quantized)`.
  That PSNR drop = the component's sensitivity `D_c(b)`.
- **Coordinate & activation conventions are settled** (they render the bonsai
  upright and in correct colour — see `renders/bonsai_orbit.png`):
  - Camera: **OpenCV convention** — camera looks along **+Z**, **+X right, +Y
    down**. `viewmat` is world→camera (4×4). `K` is a 3×3 pinhole intrinsic.
  - World **up axis = −Y** (that orientation renders the bonsai right-side-up).
  - Activations (PLY stores pre-activation values):
    - `scale`   → `exp(scale_i)`
    - `opacity` → `sigmoid(opacity)`
    - color (degree 0) → `RGB = clamp(0.5 + 0.28209479177387814 * f_dc_i, 0, 1)`
  - `quat = (rot_0, rot_1, rot_2, rot_3)` interpreted as **(w, x, y, z)**,
    normalize before use.
- Test file: `C:/Users/min/Downloads/bonsai.ply` — **degree 0** (all `f_rest_*`
  are 0), 272,956 gaussians, 62 float properties, plus 32 `+inf` opacity values
  (sanitize non-finite to 0 on load, like the codec does).

---

## 1. The full forward pipeline (what you're building)

```
per Gaussian (1 CUDA thread each) — "preprocess":
  1. transform mean world→camera→screen        (projection)
  2. frustum cull (behind camera / off-screen)
  3. build 3D covariance  Σ = R S Sᵀ Rᵀ         (from quat + scale)
  4. project to 2D covariance Σ' = J Σ_cam Jᵀ    (EWA) + dilation
  5. invert Σ' → conic; compute screen-space radius/bbox
  6. compute color (degree-0 DC → RGB)
sort:
  order gaussians (and their tile touches) by depth
per pixel (1 thread each, in 16×16 tiles) — "render":
  front→back: α = opacity·exp(-½ dᵀ conic d); C += T·α·color; T·=(1-α)
```

Build it in the staged order below, not all at once.

---

## 2. Staged build plan (do these in order)

### Stage 0 — toolchain + image output
Kernel writes a color per pixel into a device buffer; copy to host; save an image.
- Buffer: `float3*` (HDR, clamp on save) or `uchar3*` (0–255 directly). Either is fine.
- Image format: **PPM** is zero-dependency — write ASCII header `P6\n{W} {H}\n255\n`
  then raw RGB bytes. (PNG later via `stb_image_write.h`, single header, if wanted.)
- **Accept when:** a gradient/solid image saves and opens correctly.

### Stage 1 — project means only (point cloud)
Render each Gaussian as a single colored pixel at its projected screen position.
Ignore covariance and blending; just write the color (optionally z-test).
- Proves the camera math (world→cam→screen) and conventions.
- **Accept when:** the output is a recognizable bonsai-shaped point cloud,
  upright (world up = −Y).

### Stage 2 — 2D covariance (elliptical splats)
Add `Σ` from scale+quat, project to `Σ'`, invert to conic, and evaluate the
Gaussian falloff for every pixel inside each splat's bbox. Blending can still be
naive (additive) for now.
- This stage has the hard math (§3.4–3.6). Read INRIA `computeCov2D`.
- **Accept when:** splats are soft blobs, not points; shapes/orientations look right.

### Stage 3 — depth sort + alpha compositing
Sort gaussians by depth; per pixel accumulate front→back with transmittance.
- **Accept when:** correct occlusion, colours, and coverage — no additive
  saturation (front splats properly hide those behind them).

### Stage 4 — performance (only if needed)
If naive is too slow, add 16×16 tiling: per-tile gaussian lists so each pixel only
processes gaussians whose bbox touches its tile. Measurement tool ≠ real-time, so
a bbox-limited naive version may already suffice. **Correctness before speed.**

---

## 3. The math (reference formulas)

### 3.1 Load + activate (host)
Per gaussian, from the parsed columns (reuse `parse_ply_header`):
```
mean   = (x, y, z)
scale  = exp(scale_0, scale_1, scale_2)
q      = normalize(rot_0, rot_1, rot_2, rot_3)        // (w,x,y,z)
opacity= sigmoid(opacity_raw)
rgb    = clamp(0.5 + 0.2820947918 * (f_dc_0, f_dc_1, f_dc_2), 0, 1)
```
Sanitize: any non-finite input → 0.

### 3.2 Camera
Intrinsics for image W×H, vertical FOV θ:
```
fx = fy = 0.5 * W / tan(θ/2)
K = [[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]]
```
`look_at(cam_pos, target, up=(0,-1,0))` → world→camera 4×4 (OpenCV):
```
f = normalize(target - cam_pos)      // forward = +Z
r = normalize(cross(f, up))          // right   = +X
d = cross(f, r)                       // down    = +Y
R = rows [r; d; f]                    // 3×3
t = -R * cam_pos
viewmat = [[R, t], [0,0,0,1]]
```

### 3.3 Project mean (camera → screen)
```
t_cam = R_view * mean + t_view          // camera-space point
if t_cam.z <= 0: cull                    // behind camera
u = fx * t_cam.x / t_cam.z + W/2
v = fy * t_cam.y / t_cam.z + H/2
depth = t_cam.z                          // for sorting
```

### 3.4 3D covariance from scale + quaternion
```
R = quat_to_mat3(q)                      // 3×3 rotation from (w,x,y,z)
S = diag(scale.x, scale.y, scale.z)      // scale already exp-activated
M = R * S
Σ = M * Mᵀ                               // symmetric 3×3
```
`quat_to_mat3(w,x,y,z)` (normalized):
```
[[1-2(y²+z²),  2(xy-wz),   2(xz+wy)],
 [ 2(xy+wz), 1-2(x²+z²),   2(yz-wx)],
 [ 2(xz-wy),  2(yz+wx),  1-2(x²+y²)]]
```

### 3.5 EWA projection to 2D covariance
Jacobian of the perspective projection at camera-space point `t = t_cam`:
```
J = [[fx/t.z,    0,    -fx*t.x/(t.z*t.z)],
     [  0,    fy/t.z,  -fy*t.y/(t.z*t.z)]]     // 2×3
```
Rotate Σ into camera space with the view rotation `W = R_view` (3×3), then project:
```
Σ_cam = W * Σ * Wᵀ                       // 3×3
Σ_2D  = J * Σ_cam * Jᵀ                    // 2×2
```
**Low-pass dilation** (so tiny splats don't vanish between pixels):
```
Σ_2D[0][0] += 0.3
Σ_2D[1][1] += 0.3
```

### 3.6 Conic (inverse 2×2) + radius
For `Σ_2D = [[a, b], [b, c]]`:
```
det = a*c - b*b
conic = (1/det) * [c, -b, a]             // store as (a', b', c') for the quadratic form
```
Screen radius from eigenvalues (for the bbox):
```
mid = 0.5*(a + c)
λ1 = mid + sqrt(max(0.1, mid*mid - det))
λ2 = mid - sqrt(max(0.1, mid*mid - det))
radius = ceil(3 * sqrt(max(λ1, λ2)))      // 3σ footprint
bbox = [u±radius, v±radius] clamped to image
```

### 3.7 Per-pixel evaluation + compositing
For pixel `(px, py)` and a gaussian with screen mean `(u, v)`, conic `(a',b',c')`,
color, opacity:
```
dx = px - u;  dy = py - v
power = -0.5*(a'*dx*dx + c'*dy*dy) - b'*dx*dy
if power > 0: skip
α = min(0.99, opacity * exp(power))
if α < 1/255: skip
C   += T * α * color        // T = transmittance, starts at 1
T   *= (1 - α)
if T < 1e-4: pixel is done  // early terminate
```
Process gaussians **front→back** (sorted by depth). Final pixel color = `C`
(optionally over a background; `C` already has `(1−T)` coverage).

### 3.8 PSNR
Given reference image `A` and test image `B` (both float RGB in [0,1]):
```
MSE  = mean over all pixels & channels of (A - B)²
PSNR = 10 * log10(1 / MSE)     // MAX = 1;  ∞ if MSE == 0
```

---

## 4. Build setup (CMake + CUDA)

Add CUDA to the existing CMake. Sketch:
```cmake
project(plycodec LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)          # RTX 3080 = sm_86; laptop: set to its GPU
add_executable(renderer src/renderer.cu src/plyparser.cpp)
```
**Windows toolchain gotcha (hit on the desktop):** CUDA rejects too-new MSVC.
If nvcc errors "unsupported Microsoft Visual Studio version", force an older
toolset — build from a *Native Tools* prompt after
`vcvars64.bat -vcvars_ver=14.38` (or whichever ≤ VS2022 toolset is installed).
On the laptop, check `nvcc --version` vs the installed VS; use a compatible pair.

---

## 5. Verification

- **Self-reference (no external oracle):** the distortion meter compares
  `render(original)` vs `render(quantized)` from the **same** camera, so it needs
  no ground-truth image — any renderer bias cancels. Validate correctness by eye
  (`renders/bonsai_orbit.png`: upright bonsai, correct colours/occlusion), then
  trust PSNR as a relative quantization-damage measure.
- Sanity numbers: with 272,956 gaussians at 800×800, a correct render has mean
  alpha ≈ 0.4–0.5 (the scene fills roughly half the frame from orbit distance
  2.5× the p90 radius).

---

## 6. Gotchas (each has bitten someone)

1. **Covariance dilation (`+0.3`)** — omit it and tiny gaussians flicker/vanish.
2. **Convention signs** — OpenCV (+Z forward, +Y down), world up = −Y. A wrong
   sign gives a black or upside-down frame; up=+Y flips it, so keep up=−Y.
3. **Conic is the inverse** — precompute per gaussian; never invert per pixel.
4. **`power > 0` guard** — floating error can make the exponent positive; skip.
5. **Front-to-back vs back-to-front** — the compositing formula above is
   front-to-back with transmittance `T`. Be consistent with the sort order.
6. **Non-finite inputs** — sanitize on load (32 inf opacities in bonsai).

---

## 7. INRIA reference (read, then rewrite in your own code)

Original `diff-gaussian-rasterization`, `cuda_rasterizer/forward.cu`:
- `computeCov3D()` — §3.4 (scale+quat → 3D covariance)
- `computeCov2D()` — §3.5 (EWA Jacobian + dilation) — the trickiest part
- `preprocessCUDA()` — §3.3–3.6 (projection, conic, radius, tiles)
- `renderCUDA()` — §3.7 (tiled front-to-back compositing)

Understand the math, don't copy — this is the highest-value part of the project
to have written yourself.

---

## 8. Definition of done

A `renderer` executable that:
1. loads a `.ply`, renders an orbit view, saves an upright colour image;
2. exposes a function to render given (params, camera) so the ablation can call it
   with quantized params;
3. computes PSNR between two renders.

Then the ablation loop (quantize component c to b bits → render → PSNR vs base)
produces `D_c(b)`, which combined with `rd_profile`'s `H_c(b)` completes the
rate–distortion curve.
