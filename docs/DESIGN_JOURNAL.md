# 3D Gaussian Splatting PLY Codec — Design Journal

> A rate–distortion-driven compressor for 3D Gaussian Splatting scenes.
> This document records **what** we built, **why** we built it this way, what the
> **research literature** says, and what the **measurements** revealed. It is the
> running source for a later portfolio write-up.

---

## 1. The problem

A trained 3DGS scene is a list of **N Gaussians**, each carrying ~59–62 float
attributes (position, orientation, scale, opacity, view-dependent color). A
medium scene is 270k–6M Gaussians → tens to hundreds of MB of raw `.ply`. That
is too large to stream or ship.

The goal of this project is a codec that pushes a scene **far down and to the
left** on the rate–distortion (R-D) plane:

```
   PSNR (quality) ↑
        │                    ● gsplat PngCompression (the baseline to beat)
        │            ●  ← our target: same quality, smaller  (or better quality, same size)
        │
        └───────────────────────────────→  file size (rate)
```

The single scalar that scores the whole project is this curve. Everything below
serves the question *"how do I move a point down-left without wrecking the render?"*

---

## 2. The compression pipeline (mental model)

Every serious 3DGS compressor — and ours — is four stages:

```
.ply  (N Gaussians × ~59 floats)
  │
  ├─[0] Ordering      Morton / PLAS sort        put similar Gaussians next to each other
  ├─[1] Transform     log / delta / reparam /   re-express into an easy-to-code shape
  │                   PCA / autoencoder
  ├─[2] Quantization  SQ (scalar) / VQ (vector)  real numbers → integer symbols  ★ loss
  ├─[3] Entropy code  histogram+arithmetic /     symbols → bits                  ★ shrink
  │                   learned context model
  ▼
bitstream (+ optional model blob)
```

Two stages do the actual work and it helps to keep them straight:

- **[2] Quantization is where information is *lost*.** Rounding a float to an
  integer symbol is irreversible. The *rate* you pay and the *distortion* you
  incur both originate here.
- **[3] Entropy coding is where the file gets *smaller*, losslessly.** Given the
  symbols, it re-packs them near their Shannon entropy. It never loses anything
  and never *can* beat the entropy of the symbol stream it's handed.

The strategic consequence: **[3] can only exploit whatever redundancy [0] and [1]
left in the symbols.** If position coordinates arrive as near-random numbers, no
entropy coder saves you — you had to make them compressible earlier, by ordering
and delta-coding. This is the thesis the measurements later confirm.

---

## 3. What the research says (per component)

Surveyed the current 3DGS-compression literature to ground the per-attribute
decisions rather than guessing. **Every claim below was verified against the
primary source** (the paper PDF/HTML, not a secondary summary); the verification
status of each is marked. Sources §8.

### 3.1 The load-bearing paper: c3dgs (Niedermayr et al., CVPR 2024) — read in full

Most of our per-component reasoning traces to this paper, verified against its text:

- **Sensitivity is defined and measured** (their Eq. 3):
  `S(p) = (1/ΣᵢPᵢ) · Σᵢ |∂Eᵢ/∂p|` — N training images, `Pᵢ` = pixels in image
  *i*, `E` = total image energy (Σ RGB). *"a large gradient magnitude indicates
  high sensitivity."* **This is exactly the sensitivity formula in our project
  plan** — now confirmed as c3dgs's, not folklore.
- **Sensitivity is folded into the VQ objective** (their Eq. 5):
  `D(x, cₖ) = S(x)·‖x − cₖ‖²` — the codebook clusters low-sensitivity vectors
  tightly and spares high-sensitivity ones. Our plan's `D(x,c)=S(x)‖x−c‖²` is
  verbatim this.
- **Position is the most quantization-sensitive attribute — stated explicitly, not
  inferred:** *"We quantize all Gaussian parameters despite position to an 8-bit
  representation with the Min-Max scheme. 16-bit float quantization is used for
  position, as a further reduction decreases the reconstruction quality
  considerably."* → position gets 2× the bits of everything else.
- **SH and shape go to two VQ codebooks:** SH coefficients are vector-clustered;
  Gaussian shape (scale+rotation) is a second codebook, after **reparametrizing
  scale** `s = η·ŝ` (magnitude split from a normalized direction). High-sensitivity
  SH is kept out of clustering: *"only SH coefficients of a tiny fraction of all
  Gaussians (<5%)... indicate a high sensitivity."*
- **Opacity** is scalar-quantized to 8-bit **after** the sigmoid; scale/rotation
  are quantized **before** their normalization/exp activation.
- **Entropy stage = DEFLATE (LZ77 + Huffman) after Morton-order sorting** — *"The
  data is then compressed using DEFLATE... By ordering the Gaussians according to
  their positions along a Z-order curve in Morton order, the coherence can be
  exploited."* → **our milestone ⑥ (zlib/DEFLATE baseline) is literally c3dgs's
  entropy stage.**
- **Ratio: ~30×** (their Fig. 2 shows 1.4 GB → 47 MB; text says "up to 31×").

### 3.2 Per-component table (each cell tagged with its verified source)

| Component | DOF / nature | Transform [1] | Quant [2] | Verified basis |
|---|---|---|---|---|
| **position** (x,y,z) | continuous, spatial | ordering: 1D Morton→DEFLATE (c3dgs) *or* 2D PLAS→image codec (SOG) | **high-bit SQ (16-bit)** | c3dgs §4.2 (16-bit, explicit) ✅ · SOG PLAS + JPEG XL ✅ |
| **rotation** (quaternion, 4) | unit-norm → 3 DOF | quaternion → Euler | SQ (before norm activation) | MesonGS: Euler chosen to **preserve covariance positive-definiteness** under quantization, and saves 1 float ✅ · c3dgs §4.2 ✅ |
| **scale** (3) | positive; stored pre-exp so already log-domain | c3dgs: normalize `s=ηŝ`; SQ operates in log space via exp activation | SQ (before exp activation) | c3dgs §4.1–4.2 ✅ · "log-then-SQ" is a *consequence* of the exp activation, not a cited transform 🔶 |
| **opacity** (1) | scalar, post-sigmoid | — | SQ (8-bit, after sigmoid) | c3dgs §4.2 ✅ |
| **SH DC** (3, base color) | perceptually visible | — (GSICO: RGB→YUV) | SQ / VQ | c3dgs treats all SH as VQ vectors ✅ · YUV is GSICO (survey) 🔶 |
| **SH rest** (45, view-dependent) | low sensitivity, dominant size | PCA / band-prune / distill | **SQ vs VQ (to measure)** | c3dgs: <5% of SH high-sensitivity, rest clustered ✅ · LightGaussian distills SH degree (survey) 🔶 · Reduced3DGS prunes bands (survey) 🔶 |

Legend: ✅ = confirmed in the paper's own text; 🔶 = from the survey/secondary, primary not yet read.

### 3.3 Sensitivity ranking — what's proven vs inferred

```
position  ≫  { SH DC, rotation, scale, opacity }  >  SH rest
 └ c3dgs proves this (16-bit vs 8-bit)              └ c3dgs: most SH low-sensitivity
        └────────── middle order is INFERRED, not from any paper ──────────┘
```

**Honesty note:** c3dgs proves the two *ends* — position is most sensitive (gets
16-bit), and the vast majority of SH is least sensitive (clustered / prunable; up
to 15% of splats have *zero* color sensitivity and are pruned). The relative order
*within* the middle group (rotation vs scale vs opacity) is **our inference from
general practice, not a cited result.** M1's ablation (Meter B) will measure the
real ranking on our own scenes and replace this guess with data.

### 3.4 Learned entropy models (the expensive rung — HAC, ECCV 2024)

Verified against the HAC paper: it predicts, per anchor, a **Gaussian
distribution** `(μ, σ)` for each quantized attribute from a **hash-grid context**
via an MLP, then trains against an **entropy loss** `L = Σ −log₂ p(f̂)`. Base
quantization steps are set per attribute (`Q₀ = {1, 0.001, 0.2}` for
feature/scaling/offset). Reported **>75× vs vanilla 3DGS, >11× vs Scaffold-GS.**

Precision point: HAC does **not** compute mutual information as a loss term — it
minimizes entropy under a learned context. The information-theoretic ceiling
`I(component ; context)` in our plan is the *theoretical bound* on what such a
model can save; measuring `I` first (plan step ⑤) is our screen to decide whether
building HAC-style machinery is worth it for a given component. That framing is
ours; the entropy-under-context mechanism is HAC's.

---

## 4. Why the code is built the way it is

Design decisions and their reasoning — this is the section a portfolio reader
learns the most from.

### 4.1 Two-pass streaming, chunked I/O

The min/max bounds for quantization need a full look at the data *before* any
symbol can be produced. So:

```
pass 1:  read chunk → update_bounds (min/max)      ── needs all data
         compute_scales                            ── scale = max_val / range
pass 2:  read chunk → encode → accumulate histogram
```

The body is never fully resident — we stream fixed-size chunks (e.g. 65536
Gaussians). A subtle correctness point: the **final chunk is short**, so the
inner loops iterate `to_read`, never `chunk_size`, or the leftover buffer tail
injects phantom Gaussians into the histogram.

*Future optimization noted, not yet done:* pass 1 is a separate 67 MB traversal.
It can be **fused into the PLY parse** — compute bounds while the bytes are already
hot in cache — deleting an entire pass. That is worth more than any SIMD tuning of
the bounds kernel.

### 4.2 Bit depth is capped at b ≤ 16

We support `uint8_t` (b ≤ 8) and `uint16_t` (b ≤ 16) symbol storage and reject
b > 16. Three independent reasons converge on this:

1. **Float only carries ~24 bits of real precision.** Quantizing to more bins
   than the source can distinguish just creates empty bins — rate with no fidelity.
2. **A per-symbol histogram at b=32 needs 2³² bins ≈ 1 TB.** The entropy-profiling
   method is physically incompatible with high bit depth.
3. **The `int q` intermediate in `encode` overflows past ~b=30**, and `(int)max_val`
   for b=32 wraps to −1, making `std::clamp(q,0,-1)` undefined behavior.

Capping at 16 keeps the simple `int` code provably correct — no `int64`/`uint32`
machinery needed. (3DGS quantization is typically 8–12 bits anyway.)

### 4.3 Min-max SQ normalizes away *width*, so entropy measures *shape*

Because each column is rescaled to `[0, max_val]` using its own min/max, the
**width** of a distribution is divided out. Entropy then depends only on the
distribution's **shape**:

- A narrow Gaussian and a wide Gaussian → *same* entropy.
- Only genuinely **peaked / skewed** shapes (bimodal opacity, leptokurtic SH-rest)
  yield low entropy and therefore compress.

This is why "the values are small" is *not* a reason something compresses — a fact
that bit us in synthetic testing and is worth stating explicitly.

### 4.4 Aggregate entropy per *component*, not per raw column

Property names from the PLY header are mapped to semantic groups
(`component_of()`): position / sh_dc / sh_rest / opacity / scale / rotation /
normal. Bit-allocation and codec choices are made per *component*, so the profiler
sums the per-dimension entropies `H_d` into those groups.

### 4.5 `std::isfinite` guards on every bound/encode

Real 3DGS exports contain `inf`/`nan` (degenerate Gaussians). A single `inf` in a
column propagates through `std::max` and destroys that column's range. Both
`update_bounds` and `encode` skip non-finite values. (See §5 — this was found the
hard way.)

---

## 5. M0 measurements — what bonsai actually revealed

Ran the profiler on `bonsai.ply` (272,956 Gaussians, 62 dims, 67.7 MB) across a
bit sweep. The results were *more* informative than expected — they contained one
bug and three real data properties, and separating them was the whole lesson.

**Rate profile at b = 8 (after fixes):**

```
component   dims   bits/gs   ratio   reading
position     3     20.2      1.2x    continuous, near-flat
sh_dc        3     22.5      1.1x    ← pre-quantized to 8-bit in source
sh_rest     45      0.0      inf     ← all exactly zero (SH degree 0)
opacity      1      6.7      1.2x    ← 254 distinct; had 32 inf values
scale        3     19.0      1.3x    continuous
rotation     4     26.7      1.2x    continuous
normal       3      0.0      inf     ← all zero
──────────────────────────────────
TOTAL       62     95.1              entropy floor ~3.24 MB
```

### Finding 1 — this bonsai is **SH degree 0**
All 45 `f_rest_*` columns are exactly 0. They occupy 45/62 = 73% of the raw
columns and carry **zero information**. → drop them entirely; and → *this file
cannot be used to evaluate SH-rest VQ* (milestone ④ needs a degree-3 scene).

### Finding 2 — **f_dc and opacity are already 8-bit quantized in the source**
They show only 255 / 254 distinct values; their entropy caps at ~7.5 bits and
does **not grow when b goes 8 → 16**. Meanwhile position/scale/rotation *keep*
growing (position 6.7 → 14.6 bits). → give sh_dc/opacity b=8 and no more;
reserve extra bits for the genuinely-continuous components.

### Finding 3 — **order-0 entropy coding is nearly useless on the real data**
Excluding the zero columns, the 14 real dimensions compress only **1.2×** under a
histogram+arithmetic coder — because position/scale/rotation are near-flat.
This is the thesis of §2 made concrete: **the win is not in [3]; it is in [0]
ordering + [1] delta/context for position.** The literature's emphasis on spatial
context modeling is thereby validated *on our own data, before building anything*.

### The bug — 32 `inf` opacity values
Opacity first read as H = 0 (a whole dead column). Cause: 32 `+inf` values in the
source poisoned `qMax` → range = ∞ → scale = 0 → every value mapped to bin 0. Fixed
with `std::isfinite` guards. Lesson recorded as design decision §4.5.

### 5.4 Step ⑥ — DEFLATE baseline vs the entropy floor

`tools/rd_probe.py` is an **independent Python re-implementation** of the exact
min-max quantizer. It (a) cross-validates the C++ entropy — the per-component H
matched to 2 decimals (position 20.18 = 20.18, rotation 26.69 = 26.69, …),
confirming both implementations agree — and (b) measures what real **raw DEFLATE
(zlib level 9)** achieves on the identical symbols, in the planar column-contiguous
layout a real codec would store.

**Result at b = 8 — DEFLATE *equals* the order-0 floor:**

```
component  dims   H bits/gs  DEFLATE   D/H   verdict
position      3     20.18     20.28   1.01   ~order-0 floor; need ordering
sh_dc         3     22.47     22.48   1.00   ~order-0 floor
opacity       1      6.70      6.67   1.00   ~order-0 floor
scale         3     19.02     19.12   1.01   ~order-0 floor
rotation      4     26.69     27.11   1.02   ~order-0 floor
sh_rest      45      0.00      0.35    inf   empty (all zero) -> ~free
────────────────────────────────────────────
TOTAL        62     95.07     96.04   1.01
     entropy floor 3.24 MB | DEFLATE 3.28 MB | 8-bit raw 16.9 MB | orig 67.7 MB (20.9x)
```

Two conclusions, both now *measured* rather than asserted:

1. **DEFLATE ≈ H on every real component (D/H = 1.00–1.02).** LZ77's match-finder
   gets *nothing* — consecutive Gaussians in PLY order are spatially unrelated, so
   there is no repetition to exploit, and Huffman lands within 1–2 % of the
   arithmetic-coding floor. This is the plan's §4.5 "(a) ≈ (b)" case: **for order-0,
   a hand-built arithmetic coder (M2) would buy only ~1 % over stock DEFLATE.**
2. **Therefore the entire remaining opportunity is [0] ordering + [1] transform.**
   Nothing in stage [3] helps until consecutive symbols are made correlated. This
   is *why* c3dgs DEFLATEs **only after Morton sorting** — we independently measured
   the reason: without ordering, DEFLATE = the order-0 floor.

**Contrast at b = 16 (D/H = 1.19):** the pre-quantized columns (sh_dc, opacity)
jump to D/H ≈ 1.5 — byte-oriented Huffman cannot model 16-bit symbols whose true
alphabet is 8-bit, so arithmetic coding *would* win there. But b = 16 is wasteful
anyway (Finding 2). The coherent takeaway stands: **run at b = 8, where DEFLATE is
already at the floor, and spend all effort on ordering/transform.**

---

## 6. The full process (milestones)

```
M0  render-free RATE measurement            ← we are here
    ├ ① PLY parser + sanity check                                  ✅ done
    ├ ② general SQ-b quantizer + H_c(b) entropy sweep              ✅ done
    ├ ③ algebra-verdict table (assert, don't measure)             ← next (doc)
    ├ ④ SH-rest codec R-D points (k-means VQ vs PCA)              needs degree-3 ply
    ├ ⑤ learned-model ceiling: I(component ; context)            entropy-model prototype
    └ ⑥ DEFLATE baseline (zlib) vs measured H                     ✅ done (tools/rd_probe.py)

M1  SQ + PNG baseline + DISTORTION meter (gsplat render,
    held-out views, per-component ablation → sensitivity S_c)
    → combine with M0's H_c(b) → full R-D curve → bit allocation

M2  VQ + hand-built arithmetic coding on indices (the core codec)

M3  a learned part — VQ-VAE in [1] or learned entropy model in [3],
    only where ⑤'s I(c;ctx) said it pays

Extend  C++ real-time decoder; validate on garden (~6M); amortized study
```

**Two held-out axes, never mixed:**
- Meter A (rate) splits **Gaussians** — checks a model learned a rule, not memorized points.
- Meter B (distortion) splits **camera views** — checks quality loss is real, not view-overfit.

---

## 7. Traps already identified

1. **Circularity** — "render-free rate" isn't truly render-free, because making
   symbols needs a bit count and the bit count comes from sensitivity (render).
   *Fix:* make bits a free variable and sweep `H_c(b)` — precompute the whole
   rate curve; M1 later just looks up the chosen b.
2. **"Loss is only in [2]"** — false. PCA in [1] truncating 45→8 is also loss.
   Judge by degrees of freedom: numbers-in > numbers-out ⇒ lossy.
3. **VQ rate ≠ log₂K.** A K-entry codebook is used unevenly, so the true rate is
   `H(index)`, the entropy of the index distribution. Compare VQ to SQ only with
   entropy coding applied to indices.
4. **Diagnostic costing more than the codec.** Plotting k-means error to "decide
   whether to use VQ" *is* implementing VQ. Only the learned model ([3]) is
   expensive enough that measuring its ceiling first genuinely pays.
5. **Uncounted model cost.** Autoencoder / entropy-model / hash-grid weights are
   bytes in the file. Per-scene break-even: `model_bytes < N × saved_bits / 8`.

---

## 8. Sources

Verification level: **[P]** = primary source read in full and each cited claim
checked against the paper's own text; **[S]** = secondary (survey / abstract), not
yet verified against the primary.

- **[P]** c3dgs: Compressed 3D Gaussian Splatting (Niedermayr et al., CVPR 2024) — https://openaccess.thecvf.com/content/CVPR2024/papers/Niedermayr_Compressed_3D_Gaussian_Splatting_for_Accelerated_Novel_View_Synthesis_CVPR_2024_paper.pdf — sensitivity Eq. 3, VQ objective Eq. 5, position 16-bit vs 8-bit, DEFLATE+Morton, ~30× all verified.
- **[P]** MesonGS: post-training attribute transform (ECCV 2024) — https://arxiv.org/html/2409.09756v1 — quaternion→Euler (positive-definiteness), RAHT on opacity/scale/rotation/SH-DC (not scale@8-bit), LZ77 verified.
- **[P]** HAC: Hash-grid Assisted Context (ECCV 2024) — https://arxiv.org/html/2403.14530v2 — per-anchor adaptive quant, Gaussian entropy model from hash context, entropy loss, Q₀ per attribute, 75×/11× verified.
- **[P]** Self-Organizing Gaussians / PLAS (ECCV 2024) — https://arxiv.org/html/2312.13299v2 — all attributes (un-activated) → 2D grids; PLAS = Parallel Linear Assignment Sorting, arranges into a 2D grid preserving neighborhood by iterative best-match with decreasing filter size; RGB grid → lossy JPEG XL (q100), others → lossless JPEG XL; mechanism = "neighboring Gaussians have similar values, facilitating compression"; 17–42× (max 41.6× Deep Blending). All verified.
- **[S]** 3DGS Compression Survey (3dgs.zip) — https://w-m.github.io/3dgs-compression-survey/ — source for LightGaussian (SH distillation), Reduced3DGS (band prune), EAGLES (opacity), GSICO (YUV), RDO-Gaussian (scale). These attributions are **not yet primary-verified** and are marked 🔶 in §3.2.
- **[S]** Compression in 3DGS: A Survey — https://arxiv.org/pdf/2502.19457

### Verification method
Papers were read via WebFetch on the arXiv **HTML** builds (they parse far more
reliably than the PDFs) and, where no HTML build existed (c3dgs), by downloading
the PDF and reading the method pages directly. Three earlier claims were corrected
during this pass: scale "log transform" (c3dgs actually uses η-normalization; the
log behaviour is a side effect of the exp activation), quaternion→Euler rationale
(positive-definiteness, not merely saving a number), and HAC "optimizes mutual
information" (it minimizes an entropy loss under context; MI is only the
theoretical ceiling). Items still marked 🔶 / [S] are the honest remaining gap.

---

*Status: M0 steps ①, ②, ⑥ complete; ③ partially (empirically grounded now).
Key measured result: at b=8 DEFLATE already equals the order-0 entropy floor, so
the remaining win lives in [0] ordering + [1] transform, not the entropy coder.
Next: ⑤ context ceiling I(c;ctx) for position (the highest-value unknown), and a
degree-3 `.ply` to unlock ④ (SH-rest VQ). Primary-verified sources: c3dgs, HAC,
MesonGS, SOG.*