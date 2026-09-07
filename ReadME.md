# Prompt2Pixel: A From-Scratch Text-to-Image Generation System in Pure Python + NumPy — Research Project Blueprint

## TL;DR
- **Build it in two stages**: (Stage 1) a deterministic procedural renderer that parses prompt keywords and paints scenes with pure math (Perlin/simplex noise, SDFs, color theory), which produces compelling images immediately and needs zero training; (Stage 2) a small class-conditional generative model trained from scratch in NumPy on 28×28 MNIST/Fashion-MNIST, where a **Variational Autoencoder (VAE) is the recommended target** because it is the only generative family with a genuine, working, framework-free pure-NumPy reference implementation and trains in minutes-to-hours on CPU.
- **Set honest expectations**: pure-NumPy CPU training cannot produce open-vocabulary "type any prompt" photorealism. It can produce class-conditional 28×28 grayscale (or 32×32 CIFAR color, slower) samples. The strongest research angle is the bridge between the two stages: use the Stage-1 procedural renderer to synthesize a labeled training dataset, then train the Stage-2 model on it — a self-contained, download-free experiment.
- **All image I/O, math, and models are achievable with only Python stdlib + NumPy**: PNG via `zlib`+`struct`+CRC32, text via bitmap glyphs and Bresenham/SDF rasterization, autograd/layers/optimizers hand-derived. A pure-NumPy DDPM is *possible* but there is **no existing reference implementation**, so it is the ambitious "reach" deliverable, not the baseline.

## Key Findings

1. **Writing a valid PNG needs only `zlib` and `struct`.** A PNG is an 8-byte signature followed by length-prefixed chunks (IHDR, IDAT, IEND), each ending in a CRC32 over `chunk_type + data`. Python's `zlib.crc32` implements exactly PNG's CRC parameters (IEEE CRC-32, reflected polynomial 0xEDB88320, init 0xFFFFFFFF, final XOR 0xFFFFFFFF), and `zlib.compress` produces the required zlib-wrapped DEFLATE stream. Each scanline must be prefixed with a filter-type byte (0 = None is the simplest valid choice). This is ~40 lines of code.

2. **Text can be drawn without PIL** either via a hardcoded bitmap font (each glyph an array of hex/bit patterns blitted pixel-by-pixel) or via vector glyph outlines evaluated with quadratic Bézier curves + scanline polygon fill + supersampling anti-aliasing. Bresenham's integer line algorithm and the midpoint circle algorithm cover primitives.

3. **The procedural stage is mathematically rich and immediately rewarding.** Perlin noise (with the quintic fade 6t⁵−15t⁴+10t³), fractal Brownian motion, Worley noise, signed distance fields with smooth-minimum blending, and HSV↔RGB color math let you render skies, terrain, water, and clouds deterministically from a prompt hash.

4. **For the trainable model, feasibility ranks VAE > GAN > DDPM > autoregressive on pure-NumPy CPU.** A genuine framework-free pure-NumPy DDPM on MNIST does not exist in public repositories; every "DDPM from scratch" repo uses PyTorch/TF. Working pure-NumPy references exist for VAE (clean, Colab-ready) and GAN (imperfect). This strongly informs the recommendation.

5. **Text conditioning from scratch is realistic only as class conditioning or small learned embeddings**, injected via concatenation, FiLM (scale+shift), or a minimal NumPy cross-attention block. Open-vocabulary CLIP-style conditioning is out of scope without pretrained weights.

## Details

### A. Image output without libraries

**PNG (recommended primary format).** The file layout:

```
\x89PNG\r\n\x1a\n                     # 8-byte signature
[len][b'IHDR'][13 bytes data][crc]   # width,height (4B each, big-endian),
                                     # bit depth=8, color type (2=RGB,6=RGBA,0=gray),
                                     # compression=0, filter=0, interlace=0
[len][b'IDAT'][zlib(filtered rows)][crc]
[len][b'IEND'][][crc]
```

A chunk writer, verified against multiple independent tutorials (Darius' "Generating a PNG File in Python", bamfordresearch.com "One Hour PNG"):

```python
import struct, zlib
def chunk(out, ctype, data):
    out.write(struct.pack('>I', len(data)))   # 4-byte big-endian length
    out.write(ctype)                          # e.g. b'IHDR'
    out.write(data)
    crc = zlib.crc32(ctype)                    # CRC over TYPE + DATA only
    crc = zlib.crc32(data, crc) & 0xffffffff
    out.write(struct.pack('>I', crc))
```

**Key mathematical detail — the CRC32.** CRC-32 treats the byte stream as a polynomial over GF(2) and computes the remainder modulo the standard IEEE generator polynomial, stored in reflected form as **0xEDB88320** (zlib's `crc32.c` comments this literally: `#define POLY 0xedb88320 /* p(x) reflected, with x^32 implied */`). The algorithm applies an initial value of 0xFFFFFFFF and a final XOR with 0xFFFFFFFF. `zlib.crc32` matches PNG's parameters exactly, so you do *not* need to reimplement it — but for the "explain every line" deliverable you can also implement the bitwise table-driven version and verify it against `zlib.crc32(...)` as the authoritative reference (do not rely on a memorized hex constant; generate the expected value with the stdlib).

**Scanline filtering.** Before compression each row gets a leading filter byte. Filter 0 (None) stores raw bytes and always works. Filters 1–4 (Sub, Up, Average, Paeth) improve compression by predicting each byte from neighbors; the paper can present the Paeth predictor math as an optional optimization.

**PPM/BMP fallbacks (simpler, no compression).**
- **P6 binary PPM**: ASCII header `P6\n{width} {height}\n255\n` followed by raw RGB bytes. Trivial — ~5 lines.
- **BMP 24-bit**: a 14-byte BITMAPFILEHEADER + 40-byte BITMAPINFOHEADER, pixels stored **bottom-up, BGR order**, and **each row padded to a multiple of 4 bytes** (`padding = (4 - (width*3) % 4) % 4`). This padding rule is the classic gotcha to explain.

**Reading datasets back without PIL.** MNIST/Fashion-MNIST use the IDX ubyte format: a big-endian magic number (0x00000803 = 2051 for images: byte 3 = 0x08 = unsigned byte type, byte 4 = 0x03 = 3 dimensions), then count, rows, cols as 4-byte big-endian ints, then raw pixel bytes row-major. The canonical loader:
```python
with open(images_path,'rb') as f:
    magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows*cols)
```
CIFAR-10 ships as Python `pickle` files (pickle is stdlib) — `pickle.load(f, encoding='bytes')` yields a dict with a 10000×3072 uint8 array (channels R,G,B each 1024, row-major).

### B. Font rasterization / drawing primitives

**Bitmap font.** Define each glyph as a small array of bytes where each bit = one pixel (e.g. an 8×8 glyph is 8 bytes; bit `j` of byte `i` sets pixel (i,j)). Blitting is: for each set bit, write the text color into the pixel buffer. This is fully deterministic and the simplest path to "text on the image."

**Vector font (advanced).** Represent glyph outlines as closed contours of quadratic Bézier segments **B(t) = (1−t)²P₀ + 2(1−t)t P₁ + t²P₂**, flatten each curve into line segments by sampling t, then fill using the **scanline polygon fill**: for each horizontal line y, find all edge intersections, sort x-values, and fill spans. Use the **even-odd** or **nonzero winding** rule to decide inside/outside (winding handles overlapping contours and holes correctly). Anti-alias by rendering at 3–4× resolution and box-downsampling (supersampling).

**Primitives.**
- **Bresenham line**: integer-only; tracks an error term incremented by Δy each x-step, and when error > ½ (implemented via integer `err -= dy; if err<0: y+=sy; err+=dx`) steps y. No floating point.
- **Midpoint circle**: uses a decision parameter to pick between E and NE pixels using 8-way symmetry.
- **Alpha blending (the "over" operator)**: `out = src·α + dst·(1−α)` per channel; premultiplied form `out = src + dst·(1−α)`. This is the compositing math for layering the procedural scene and text.

### C. Procedural / algorithmic text-to-image (Stage 1)

**Prompt → deterministic seed.** Hash the prompt string (`hashlib` or a simple polynomial rolling hash) to an integer and seed `np.random.default_rng(seed)`. Same prompt → same image, which is essential for reproducibility and for the paper's experiments.

**Prompt parsing without NLP libs.** Lowercase, split on whitespace/punctuation, strip a small stop-word list, apply crude suffix-stripping stemming (drop `-s`, `-ing`, `-ed`), then map keywords to attributes via a dictionary: `{"sunset": warm_palette, "ocean": water_layer+blue_palette, "mountain": ridged_fBm_terrain, "night": dark_palette+stars}`. Bag-of-words counts drive layer weights; a simple slot-filling grammar (`[time] [subject] [style]`) sets global parameters.

**Perlin noise — full derivation.** On an integer lattice assign each corner a pseudorandom gradient vector **g**. For a query point, compute offset vectors from each corner, take dot products nᵢ = gᵢ·dᵢ, then interpolate with the **quintic fade** ψ(t)=6t⁵−15t⁴+10t³ (chosen over the cubic 3t²−2t³ because ψ has zero *first and second* derivatives at t=0,1, giving C² continuity and removing the visible creases/second-derivative discontinuities at cell boundaries). In 2D: bilinearly blend the four corner dot products using u=ψ(fx), v=ψ(fy). The derivative ψ′(t)=30t⁴−60t³+30t² is available in closed form for analytic normals.

**fBm (fractal Brownian motion).** Sum octaves of noise: `value = Σ amplitudeᵢ · noise(frequencyᵢ · p)`, where frequencyᵢ = lacunarity^i (typically 2) and amplitudeᵢ = persistence^i (typically 0.5). More octaves = more fine detail. This generates natural-looking terrain heightmaps and cloud density.

**Value noise & Worley noise.** Value noise interpolates random *values* (not gradients) at lattice points — cheaper, slightly blockier. Worley/cellular noise computes distance to the nearest of a set of random feature points, giving Voronoi-like cellular patterns (stone, water caustics).

**Signed distance fields (SDFs).** A shape is a function returning signed distance to its surface (negative inside). Primitives (Inigo Quilez's canonical formulas):
- Circle: `sdCircle(p, r) = length(p) − r`
- Box: `sdBox(p, b) = length(max(|p|−b, 0)) + min(max(dx,dy),0)`
- **Smooth minimum** (blends shapes organically): `smin(a,b,k) = mix(b,a,h) − k·h·(1−h)` where `h = clamp(0.5 + 0.5·(b−a)/k, 0, 1)`. The parameter k controls blend width.
Compose scenes by taking `min` of SDFs (union), `max` of negated (subtraction), and smin for organic joins. **Raymarching** renders 3D-looking SDF scenes: from each pixel's camera ray, repeatedly step forward by the SDF value (a safe distance guaranteeing no overshoot) until distance ≈ 0 (hit) or a max distance (miss).

**Color theory math.**
- **HSV→RGB**: given H∈[0,360), S,V∈[0,1]: C=V·S, X=C·(1−|((H/60) mod 2)−1|), m=V−C, pick (R′,G′,B′) by hextant, add m.
- **RGB→HSV**: from max/min of channels for V, chroma, and hue.
- **Gradient interpolation**: linear or smoothstep blend between palette colors along a coordinate.
- **Palette from prompt**: map sentiment/keywords to base hue + harmony rule (complementary = +180°, analogous = ±30°, triadic = ±120°).
- **Gamma correction**: display-linear conversion `out = in^(1/2.2)` (encode) so brightness looks perceptually correct.

**Scene compositing.** Build layers back-to-front — sky gradient → distant fBm mountains with depth fog `color = mix(scene, fog, 1−exp(−density·depth))` → midground → water (reflect + ripple noise) → clouds (fBm density as alpha) — combining with the alpha-over operator. Depth-based fog gives aerial perspective.

### D. Neural network from scratch in NumPy (Stage 2 foundations)

**Dense layer.** Forward: `z = xW + b`. Backprop via chain rule: given upstream `dz`, `dW = xᵀ dz`, `db = Σ dz`, `dx = dz Wᵀ`.

**Activations & derivatives.**
- ReLU: `max(0,x)`, derivative 1 if x>0 else 0.
- LeakyReLU: `x if x>0 else αx`, derivative 1 or α.
- Sigmoid σ: derivative σ(1−σ).
- tanh: derivative 1−tanh².
- GELU: `x·Φ(x)` (Φ = standard normal CDF), with the tanh approximation for cheap eval.
- SiLU/Swish: `x·σ(x)`, derivative σ(x)+x·σ(x)(1−σ(x)).

**Softmax + cross-entropy.** The famous simplification: for softmax output p and one-hot label y, the gradient w.r.t. logits is simply **`dz = p − y`**. Derive it via the quotient rule on softmax combined with the log-loss to show the cancellation.

**MSE loss.** `L = ½‖ŷ−y‖²`, gradient `ŷ−y`.

**Convolution via im2col.** Unfold each sliding window into a column, stack into a matrix `X_col`, reshape kernels to rows `W_row`, then `out = W_row · X_col` (a single GEMM). Output size `out = 1 + (H + 2P − FH)/S`. Backward: `dW` from `dout · X_colᵀ`, and `dX` via **col2im** (the inverse folding, scattering-adding overlapping gradients). Transposed convolution (for upsampling in the decoder/U-Net) is the gradient of convolution — equivalently insert zeros between inputs (stride) and convolve.

**Normalization layers.**
- **BatchNorm** forward: `x̂ = (x−μ_B)/√(σ²_B+ε)`, `y = γx̂+β`. Backward (the simplified closed form): `dβ=Σdy`, `dγ=Σ dy·x̂`, and `dx = (1/(N·√(σ²+ε)))·(N·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂))` where `dx̂ = dy·γ`.
- **LayerNorm**: normalize over features per-sample (no batch dependence) — better for small batches and attention.
- **GroupNorm**: normalize over channel groups — good for conv nets with tiny batches.

**Optimizers (derived).**
- SGD: `θ ← θ − η·g`.
- Momentum: `v ← μv + g; θ ← θ − ηv`.
- RMSProp: `s ← βs + (1−β)g²; θ ← θ − η·g/√(s+ε)`.
- **Adam** (Kingma & Ba 2014): `m ← β₁m+(1−β₁)g`, `v ← β₂v+(1−β₂)g²`, **bias-correct** `m̂=m/(1−β₁ᵗ)`, `v̂=v/(1−β₂ᵗ)`, update `θ ← θ − η·m̂/(√v̂+ε)`. The bias correction exists because m,v initialize at 0 and are biased toward 0 in early steps; dividing by (1−βᵗ) undoes this. Defaults β₁=0.9, β₂=0.999, ε=1e-8, η=1e-3.
- **AdamW**: decouples weight decay (`θ ← θ − η(m̂/(√v̂+ε) + λθ)`).
- **Schedules**: cosine decay `η_t = η_min + ½(η_max−η_min)(1+cos(πt/T))`; linear warmup for the first few hundred steps.

**Weight init (variance-preservation argument).** To keep activation variance ≈1 through depth, for a layer with fan_in inputs you need Var(W)·fan_in ≈ 1.
- **Xavier/Glorot** (tanh/sigmoid, symmetric): `Var(W)=2/(fan_in+fan_out)`, or uniform `U[−√(6/(fan_in+fan_out)), +√(...)]`.
- **He/Kaiming** (ReLU): ReLU zeros half the inputs, halving variance, so *double* it: `Var(W)=2/fan_in`, i.e. `W ~ N(0, √(2/fan_in))`. This is why deep ReLU nets need He init to avoid exponential signal decay.

**Numerical stability.** log-sum-exp trick (subtract max before exp in softmax/log-likelihoods); ε in every denominator; gradient clipping by global norm (`g ← g·min(1, clip/‖g‖)`).

### E. The generative model itself (Stage 2)

**DDPM — full math.**
- Forward noising: `q(x_t|x_{t−1}) = N(x_t; √(1−β_t)x_{t−1}, β_t I)`. The √(1−β_t) scaling keeps variance ≈1.
- Closed-form jump to any t (reparameterization): with ᾱ_t = Π(1−β_i), **x_t = √(ᾱ_t)·x₀ + √(1−ᾱ_t)·ε**, ε~N(0,I). This O(1) sampling is what makes training tractable.
- The variational bound reduces (assuming Gaussian posteriors with fixed variance) to matching the forward posterior mean; reparameterizing to predict noise gives the **simplified loss** `L_simple = E_{t,x₀,ε}[‖ε − ε_θ(x_t, t)‖²]` — a plain MSE between true and predicted noise. This empirically beats the full variational bound.
- Reverse mean: `μ_θ(x_t,t) = (1/√α_t)·(x_t − (β_t/√(1−ᾱ_t))·ε_θ(x_t,t))`.
- Beta schedules: **linear** — Ho, Jain & Abbeel (arXiv:2006.11239) set β linearly from β₁=1e-4 to β_T=0.02 over T=1000 steps — and **cosine** (Nichol & Dhariwal 2021): ᾱ_t = f(t)/f(0) with **f(t)=cos²(((t/T+s)/(1+s))·π/2)** and offset **s=0.008**. The cosine schedule was introduced because the linear schedule destroys information too quickly late in the process; s prevents β_t being too tiny near t=0 (chosen so √β_0 is just below the pixel bin size 1/127.5).
- **DDIM deterministic sampling** (Song, Meng & Ermon 2021): `x_{t−1} = √(ᾱ_{t−1})·x̂₀ + √(1−ᾱ_{t−1})·ε_θ(x_t,t)` where `x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ)/√(ᾱ_t)`. This deterministic (η=0) form lets you skip timesteps — 50 DDIM steps ≈ 1000 DDPM steps — a critical speedup for slow CPU sampling. (Note the notation trap: the DDIM paper's "α_t" is the *cumulative* product, i.e. everyone else's ᾱ_t.)
- **Classifier-free guidance** (Ho & Salimans 2022): train the same net with the condition randomly dropped to a null token, then at sampling extrapolate: **ε̃ = ε_θ(x_t,∅) + w·(ε_θ(x_t,c) − ε_θ(x_t,∅))**. Larger w = stronger prompt adherence, less diversity. On the drop probability, Ho & Salimans report that "p_uncond of 0.1 and 0.2 performed about equally well, but 0.5 was the worst" — so use ~0.1–0.2.

**VAE — full math (RECOMMENDED baseline).**
- ELBO: `log p(x) ≥ E_{q(z|x)}[log p(x|z)] − D_KL(q(z|x)‖p(z))`. First term = reconstruction; second = regularization toward prior N(0,I).
- **Reparameterization trick**: sample `z = μ + σ⊙ε`, ε~N(0,I), so gradients flow through μ,σ (the stochasticity is pushed into ε).
- **Closed-form Gaussian KL** (encoder N(μ,σ²) vs prior N(0,I)): **D_KL = −½ Σ(1 + log σ²ᵢ − μ²ᵢ − σ²ᵢ)**. This is why Gaussians are chosen — the KL has an exact, differentiable form.
- Conditional VAE: concatenate the class label (one-hot) to both encoder input and decoder latent.

**GAN.** Minimax `min_G max_D E[log D(x)] + E[log(1−D(G(z)))]`. Generator often uses the non-saturating loss `max E[log D(G(z))]` for better gradients. Mode collapse (generator maps many z to few outputs) and unstable adversarial dynamics make GANs hard to train in pure NumPy on CPU without tricks.

**Autoregressive (PixelCNN).** Model `p(x) = Π p(xᵢ | x_{<i})` with masked convolutions. Conceptually clean but sampling is sequential (pixel-by-pixel) and slow, and masked-conv bookkeeping is fiddly.

**Recommendation & feasibility.** For pure-NumPy CPU:
1. **VAE is the recommended baseline** — a clean, working, framework-free pure-NumPy VAE-on-MNIST reference exists (pometa0507's Variational-Autoencoder-Numpy). Single forward pass to sample; trains on CPU in minutes to a couple of hours for recognizable 28×28 digits.
2. **GAN is the second option** — pure-NumPy MNIST GANs exist (longenbach's uses NumPy for all math, PyTorch only to load data; apoorva-21's is framework-free but "still debugging"). Results are "so-so"; instability is the main risk.
3. **DDPM is the ambitious reach goal** — mathematically the most interesting and the natural target for a text-to-image paper, but **no framework-free pure-NumPy DDPM reference exists**, so you would be writing the U-Net forward/backward and diffusion loop entirely from scratch. Diffusion is inherently expensive: the DDIM paper states plainly that "it takes around 20 hours to sample 50k images of size 32 × 32 from a DDPM, but less than a minute to do so from a GAN on a Nvidia 2080 Ti GPU" — and that is on a GPU. On pure-NumPy CPU, expect long training; use DDIM few-step sampling and a *tiny* denoiser (a small MLP or 2-level U-Net) on 28×28.
- **Realistic resolutions**: 28×28 grayscale (MNIST/Fashion-MNIST) is the sweet spot; 32×32 RGB (CIFAR-10) is possible but ~3–4× slower and blurrier.

### F. Text conditioning from scratch

- **Class conditioning** (most realistic): one-hot label → embedding, injected by concatenation or FiLM.
- **Learned embedding matrix**: `E ∈ R^{vocab×d}`, look up token rows; train end-to-end.
- **Bag-of-words / character-level embeddings**: average token embeddings for a whole-prompt vector.
- **Minimal transformer encoder in NumPy**: scaled dot-product attention **Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V** (the 1/√d_k scaling prevents large dot products from pushing softmax into vanishing-gradient saturation). Multi-head: run h attentions on projected subspaces, concat, project by Wᴼ. **Positional encoding**: PE(pos,2k)=sin(pos/10000^{2k/d}), PE(pos,2k+1)=cos(pos/10000^{2k/d}). The attention backward pass is standard matmul + softmax-Jacobian backprop.
- **Injection into a denoiser U-Net**: (a) concatenate condition to input channels; (b) **FiLM / adaptive norm**: predict per-channel scale γ and shift β from the condition and apply `γ⊙h+β` after normalization; (c) cross-attention (query=image features, key/value=text tokens). **Timestep embedding**: sinusoidal embedding of t (same sin/cos formula) → small MLP → added/FiLM'd into each block.
- **Honest scope**: from-scratch conditioning works for a *small, closed* label vocabulary (digit classes, clothing classes). It cannot do open-vocabulary natural-language prompts. **The bridge experiment**: have Stage-1 render a labeled synthetic dataset (e.g., "red circle top-left", "blue box center") from templated prompts, then train Stage-2 conditioned on those structured attributes — a genuinely novel, self-contained research contribution.

### G. Datasets loadable without packages
- **MNIST / Fashion-MNIST**: IDX ubyte via `struct` + `np.frombuffer` (offset 16 for images, 8 for labels).
- **CIFAR-10**: `pickle` (stdlib) → 10000×3072 uint8 arrays.
- **Self-generated synthetic**: the Stage-1 renderer emits (image, label/attribute) pairs — no download, infinite data, and full control over the label distribution. **Recommended** for the conditioning experiments.

### H. Evaluation metrics in NumPy
- **MSE**: mean squared pixel error.
- **PSNR** = 10·log₁₀(MAX²/MSE) (MAX=255 or 1.0).
- **SSIM** (Wang et al. 2004): per-window **SSIM = [(2μ_xμ_y+C₁)(2σ_xy+C₂)] / [(μ_x²+μ_y²+C₁)(σ_x²+σ_y²+C₂)]** with C₁=(K₁L)², C₂=(K₂L)², L=dynamic range, K₁=0.01, K₂=0.03; combines luminance, contrast, structure; range [−1,1].
- **FID caveat**: true FID needs a pretrained Inception network (banned). Honestly state this and instead report a **proxy**: Fréchet distance between Gaussians fit to raw-pixel or PCA feature statistics, or feature stats from your *own* trained classifier. Note it is not comparable to published FID.
- **Diversity**: mean pairwise distance among samples; **memorization check**: nearest-neighbor (L2) of each generated sample to the training set — if too close, the model is copying.

### I. Research project packaging

**Project name options (with rationale):**
1. **PixelForge** — evokes forging pixels from raw math; clean, memorable. *(recommended)*
2. **NumPaint** — "NumPy + paint"; signals the pure-NumPy constraint and the rendering angle.
3. **fromscratch-t2i** — literal, discoverable on GitHub for the "from scratch" audience.
4. **ScratchDiffusion** — emphasizes the diffusion reach-goal.
5. **PromptCanvas** — user-facing framing (prompt → canvas).
6. **BareMetal Images** / **NoImport** — plays on the zero-dependency ethos.
7. **Prakāśa** (Sanskrit "light/manifestation") — a nod to the author's Chennai base; distinctive.

**Recommended repository structure:**
```
pixelforge/
  io/            png.py  ppm.py  bmp.py  idx_loader.py  cifar_loader.py
  draw/          primitives.py (bresenham, circle, fill)  font_bitmap.py  font_vector.py  blend.py
  noise/         perlin.py  simplex.py  value.py  worley.py  fbm.py
  render/        sdf.py  raymarch.py  color.py  scene.py  compositor.py
  prompt/        tokenizer.py  keyword_map.py  seed.py
  nn/            tensor_ops.py  layers.py  activations.py  conv.py  norm.py  losses.py  init.py  optim.py
  models/        vae.py  gan.py  ddpm.py  unet.py  attention.py  embeddings.py
  train/         train_vae.py  train_ddpm.py  data_synth.py (Stage-1 → dataset)
  sample/        sampler_ddpm.py  sampler_ddim.py  guidance.py
  eval/          metrics.py (mse, psnr, ssim, proxy_fid, nn_memorization)
  cli.py         # `python -m pixelforge "a sunset over mountains" --stage 1 --out out.png`
  paper/         main.tex  figures/
  tests/         test_png.py (assert against zlib), test_conv.py (numerical grad check), ...
```

**Research-paper structure (what goes in each section for this topic):**
- **Abstract**: the zero-dependency constraint, two-stage design, and the synthetic-data bridge as the contribution.
- **Introduction**: motivation (understanding every layer of the T2I stack), the "pure math" thesis, contributions list.
- **Related Work**: DDPM, DDIM, improved/cosine, CFG, LDM/Stable Diffusion, U-Net, transformers, CLIP, GAN, VAE, score-based models, Perlin noise; and educational from-scratch efforts (micrograd, tinygrad, annotated diffusion).
- **Method**: PNG/rasterization math; procedural pipeline; NumPy autograd & layers; the generative model; conditioning; the Stage-1→Stage-2 synthetic-data protocol.
- **Experiments**: datasets (MNIST/Fashion/CIFAR/synthetic), architectures, hyperparameters, CPU compute budget.
- **Results**: samples, loss curves, MSE/PSNR/SSIM/proxy-FID, memorization/diversity, ablations (linear vs cosine schedule, guidance weight w, VAE vs GAN).
- **Limitations**: resolution, CPU time, no open-vocabulary prompts, proxy-FID caveat.
- **Conclusion + Future Work**: GPU port, larger resolution, real text encoder.
- **References**: the citation list below.

**Key papers to cite (verified titles/authors/years):**
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models," 2020 (arXiv:2006.11239).
- Song, Meng, Ermon, "Denoising Diffusion Implicit Models," 2021 (arXiv:2010.02502).
- Nichol, Dhariwal, "Improved Denoising Diffusion Probabilistic Models," 2021 (arXiv:2102.09672).
- Ho, Salimans, "Classifier-Free Diffusion Guidance," 2022 (arXiv:2207.12598).
- Rombach, Blattmann, Lorenz, Esser, Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models," 2022.
- Ronneberger, Fischer, Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015.
- Vaswani et al., "Attention Is All You Need," 2017.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," 2021.
- Goodfellow et al., "Generative Adversarial Networks," 2014.
- Kingma, Welling, "Auto-Encoding Variational Bayes (VAE)," 2013.
- Perlin, "An Image Synthesizer," 1985; Perlin, "Improving Noise," 2002.
- Kingma, Ba, "Adam: A Method for Stochastic Optimization," 2014 (arXiv:1412.6980).
- Ioffe, Szegedy, "Batch Normalization," 2015; Ba, Kiros, Hinton, "Layer Normalization," 2016.
- He et al., "Delving Deep into Rectifiers (He init)," 2015; Glorot, Bengio, "Understanding the difficulty of training deep feedforward networks (Xavier init)," 2010.
- Song, Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution (score-based)," 2019.

**Where to publish/showcase**: arXiv (cs.LG/cs.CV) for the writeup; a well-documented GitHub repo (the real deliverable for this audience); a technical blog series (the "explain every line" content fits perfectly); and ML education workshops or community showcases.

### J. Existing from-scratch implementations to learn from
- **karpathy/micrograd** (github.com/karpathy/micrograd) — ~150-line scalar autograd + tiny NN; the canonical "understand backprop" reference. Educational-quality, highly reliable.
- **tinygrad** (github.com/tinygrad/tinygrad) — NumPy-backed tensor autograd "between PyTorch and micrograd"; good for autograd design patterns.
- **The Annotated Diffusion Model** (huggingface.co/blog/annotated-diffusion) — step-by-step DDPM with the cosine-schedule code (PyTorch, but the math/schedule ports directly to NumPy). Reliable.
- **Lilian Weng, "What are Diffusion Models?"** (lilianweng.github.io) — the standard math reference for the forward/reverse/loss derivations.
- **Pure-NumPy generative refs**: pometa0507/Variational-Autoencoder-Numpy (clean VAE-on-MNIST, Colab-ready, framework-free); longenbach/GANs_PyTorch (NumPy GAN, PyTorch only for data loading); apoorva-21/numpy-MNIST-GAN (framework-free GAN, "still debugging").
- **NN from scratch**: KDnuggets "Nothing but NumPy," the CS231n assignments (im2col conv, batchnorm backward), and Kevin Zakka's/Kratzert's batchnorm-backward derivations.
- **PNG**: Darius' "Generating a PNG File in Python," bamfordresearch "One Hour PNG," pyokagan PNG decoder.
- **Noise/SDF**: Inigo Quilez (iquilezles.org) for SDF primitives, smin, and raymarching; Scratchapixel and adrianb.io for Perlin derivations.

## Recommendations

**Phased roadmap (staged, with go/no-go benchmarks):**

1. **Phase 1 — Procedural renderer + PNG I/O (weeks 1–3).** Build `png.py` first and verify CRC against `zlib.crc32`. Then Perlin/fBm, SDFs, color, compositing, bitmap-font text. **Benchmark to proceed**: prompt → recognizable scene PNG with a text caption, fully deterministic per prompt. This alone satisfies the user's core "prompt → image without packages" goal and de-risks the whole project.

2. **Phase 2 — Synthetic dataset generation (week 4).** Use Phase-1 to emit thousands of (image, structured-label) pairs from templated prompts. This is the research hook and removes any download dependency.

3. **Phase 3 — NumPy autograd + layers (weeks 5–7).** Implement dense/conv/norm/activations/losses/optimizers with **numerical gradient checks** on every layer (`|analytic − numerical| < 1e-6`). **Benchmark**: train a small classifier to >95% on MNIST — proves the engine works before attempting generation.

4. **Phase 4 — Train the VAE baseline (weeks 8–9).** Recommended first generative model (working reference exists, CPU-friendly). **Benchmark**: recognizable class-conditional 28×28 samples; KL and reconstruction curves both decreasing.

5. **Phase 5 — Conditioning + (reach) diffusion (weeks 10–13).** Add FiLM/embedding conditioning to the VAE; then, if time allows, attempt the pure-NumPy DDPM with a tiny denoiser and DDIM 50-step sampling. **Threshold to attempt DDPM**: only if Phase-3 conv/backward is fast enough to do ~10k gradient steps overnight on your CPU; otherwise stay with VAE/GAN and present DDPM as fully-derived-but-future-work.

6. **Phase 6 — Evaluation + paper (weeks 14–16).** MSE/PSNR/SSIM/proxy-FID, diversity, memorization; ablations (cosine vs linear, guidance w); write the paper and polish the repo.

**What would change the plan**: if CPU conv is too slow (a single epoch of MNIST conv training takes many minutes), drop conv layers for a dense-only VAE/MLP-denoiser at 28×28; if you gain GPU access, the same NumPy code can be ported to CuPy (drop-in) — but that breaks the "NumPy only" purity, so keep it as a separate branch.

## Caveats
- **No open-vocabulary prompts.** From-scratch, CPU-only conditioning is limited to small closed label sets. Frame the system honestly as class/attribute-conditional, not "type any sentence."
- **A pure-NumPy DDPM has no existing reference** and is compute-heavy on CPU. Treat it as the ambitious extension; the VAE is the dependable baseline.
- **Source-quality notes**: the DDPM/DDIM/cosine/CFG/Adam/attention/SSIM/He-init formulas were cross-checked against primary arXiv papers and multiple independent secondary sources and agree verbatim. Watch the **α vs ᾱ notation trap** — the DDIM paper writes α_t for what DDPM calls the cumulative product ᾱ_t; your code must use the cumulative product where the DDIM paper writes α. Several "from scratch" repo titles mean "no diffusion library," not "no framework" — verify before relying on them.
- **CPU time is the binding constraint.** All time estimates are order-of-magnitude; profile early. Use small models, subsampled datasets during development, and DDIM few-step sampling.