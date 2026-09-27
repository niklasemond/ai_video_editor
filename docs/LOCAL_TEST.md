# Local character-replacement tests — in progress

This is an experimental staged runner, not a demonstrated replacement product.
The historical Spielberg assessment remains in FEASIBILITY.md. Its missing modules
were static findings; its memory/storage numbers were estimates, not benchmarks.

## Verified setup

- Local execution now works. Native arm64 Python 3.12, macOS 14.5, 16 GiB RAM.
- macOS reported about 112 GB available for important usage, versus 25 GB
  currently unallocated. A `df`-only capacity check was misleading for this APFS
  volume. The supervisor checks the Foundation capacity API and preserves 30 GB.
- Baseline system swap was about 3.2 GB, memory pressure warning level 2.
- PyTorch 2.5.1 passes tiny MPS FP16 and FP32 matrix tests outside the execution
  sandbox. MPS discovery inside that sandbox returns unavailable.
- Latest ComfyUI at 79be670e2d9be63e238785af307369d2b9039ed1 failed importing
  comfy-kitchen with PyTorch 2.5.1 (`list[int]` custom-op schema incompatibility).
  Pinned ComfyUI v0.3.60 plus the pinned GGUF loader imports successfully.
- Exact code revisions and model revisions, filenames, byte counts and SHA-256
  values are in the lock files. Downloaded model hashes verified against their pinned manifests.
  VACE GGUF reports architecture `wan`, 1263 tensors and 438 VACE tensors, but
  subsequent full comparison exposed a missing required tensor (see below).
- Dependency consistency check passed. Twenty custom unit tests passed, covering
  missing/corrupt weights, interrupted/resumed downloads, resource thresholds,
  actual child-process cancellation/failure/recovery, incomplete VACE keys,
  video-batch flattening and exact preservation of pixels outside the composite mask.

## First runtime failure and bounded remedy

The first CPU text-encoding attempt was stopped by the resource supervisor:
system swap rose from 3.21 GB to 8.15 GB during UMT5 loading. No diffusion sampling
ran in that attempt. Pressure recovered to normal, but the swap-growth threshold
was exceeded. These are system-wide samples, not isolated model-memory figures.

ComfyUI-GGUF eagerly dequantizes the (256384,4096) token embedding into FP16.
The adapter in `scripts/gguf_memory.py` uses the same upstream operation in
256-row chunks, retaining the same final FP16 table and bounding intermediates.
No upstream files were modified. An initial NumPy implementation differed by
up to 0.0625 and was rejected. The retained PyTorch implementation was bit-exact
against the upstream implementation on four-row samples at rows 0, 128000 and
256380. Its full text-encoding retry completed in 78.67 seconds, with finite
positive/negative embeddings. Sampled peak RSS was 3.88 GB; memory pressure stayed
normal and system swap fell slightly. CPU VAE preprocessing completed in 83.36
seconds, sampled peak RSS 3.84 GB, normal pressure.

## Community checkpoint defect and exact recovery

The first diffusion load silently selected plain WAN21 and discarded all VACE
controls because the community Q8 checkpoint lacks `vace_patch_embedding.weight`.
The run was terminated immediately after this warning was inspected. No result
from that invalid configuration is accepted as a replacement.

The pinned complete Comfy-Org FP16 checkpoint was downloaded and its published
SHA-256 verified. It contains 1264 tensors. After removing its standard key
prefix, exactly one tensor is missing from the community GGUF and there are no
extra keys. All 797 unquantized GGUF tensors match the complete checkpoint
exactly at FP16. The missing tensor has shape (1536,96,1,2,2).

`scripts/vace_checkpoint.py` repeats the key and value comparisons and reads
only that missing tensor from the complete checkpoint. It does not invent weights,
retrain, convert the full model or alter the downloaded GGUF. The repaired loader
now identifies WAN21_Vace with no discarded conditioning tensors. A type check
refuses ordinary Wan before sampling. Corrected sampling advanced two steps at
about 18 seconds/step, then UniPC hit an unsupported MPS linear solve. A tiny
operation test verified `PYTORCH_ENABLE_MPS_FALLBACK=1` produces the expected
solution on CPU. The runner enables this local fallback; it does not change
memory protection settings or introduce cloud execution. The full 12-step retry
completed in 256.54 seconds with normal pressure throughout and slightly declining
swap. The decoder initially exposed a missing video-batch flatten operation in
the harness; matching the native VAEDecode behavior fixed it. The saved latent
was decoded without resampling, in 68.93 seconds.

## First visual result: failed

The 25-frame diagnostic output decodes and encodes as a playable 640x360 H.264
preview with AAC audio. Both streams start at zero and last exactly one second.
Representative frames show retained source shirt/tie, inadequate reference
likeness and colored body artifacts. This is a quality failure, not success.

The bundled native inpainting template blanks the masked source region before
VACE encoding. The first diagnostic retained those source pixels as a reactive
control. A second bounded test follows the template's blanking behavior and uses
its non-distilled settings of 20 steps and CFG 6. Preprocessing completed in
78.01 seconds; sampling completed in 395.46 seconds and decoding in 65.4 seconds.
The output changed clothing but produced a posterized face and colored body artifacts.
A final controlled comparison changed only UniPC to Euler: sampling 402.36 seconds,
decoding 58.44 seconds. Euler improved clothing/body rendering but still had wrong
identity, colored artifacts and pose drift. All three are quality failures.
No full-clip run or browser interface is justified by these results.

## Offline and resource boundaries

Inference stages execute under a separate macOS sandbox profile denying network
access; this does not disable the Mac's network or change memory protections.
An explicit loopback socket probe under the profile returned EPERM. HF/Transformers
offline flags are also set, and API nodes are disabled. Text
encoding, source/reference VAE encoding, sampling and decoding run in separate
processes. CPU/GPU placement shares the same physical memory.

The supervisor terminates the process group on three consecutive critical
pressure samples (10 seconds apart), more than 2 GB swap growth over a minute,
or less than 30 GB available disk capacity. Child-process cancellation, failure
exit propagation and recovery pass tests; model-level recovery remains experimental.
No validated Quality preset or browser
interface is available; those follow a successful visual replacement test.

## Filtering, model terms and attribution

No prompt blacklist, NSFW classifier or external moderation call was found in
the inspected native Wan generation, Wan text-encoder and GGUF paths. This is
a bounded source inspection, not a claim about all transitive dependencies or
learned behavior. No filtering was added. Optional ComfyUI API nodes are disabled.
Both original Wan model cards additionally state restrictions concerning unlawful
or harmful use, harmful disclosure of personal information, misinformation and
targeting vulnerable populations. Those published terms are distinct from
executable filters; no unrestricted model capability is claimed.

ComfyUI is GPL-3.0; ComfyUI-GGUF is Apache-2.0. Their source and license files
remain in private dependency checkouts. Model cards attribute the VACE GGUF to
Wan-AI/Wan2.1-VACE-1.3B, UMT5 to google/umt5-xxl and the VAE to the Comfy-Org Wan
repackaging. These assets declare Apache-2.0. Community conversion provenance
is publisher-stated; hashes establish file integrity, not independent training
provenance. Model limitations still include identity drift, changed anatomy,
motion, occlusion and temporal inconsistency.

The second candidate's pinned public metadata is
QuantStack/Wan2.2-Animate-14B-GGUF at
33c51bb84d4e70ffc0d088aeb6068d40d9446fa3: Q2_K 6,457,431,872 bytes and Q3_K_S
7,969,675,072 bytes. Its native node supports reference, pose, face, background
and character masks. It shares the now-working UMT5/VAE stages. Q2_K and
CLIP Vision H were individually downloaded and hash-verified for a bounded test.
The Q2 file contains 1441 tensors, including face/pose modules, with architecture
`wan`; native loader compatibility is checked during actual loading. BF16 storage
is decoded through the GGUF loader into FP32/FP16, not executed as native BF16.
No 14B memory-fit claim is made before sampling.

The Animate test uses 13 frames at 192x256, Euler, 20 steps, CFG 5 and the same
reference/masked source segment. Pinned DWPose ONNX runs on CPU with a manually
selected performer box from each frame mask. Source-derived pose and 512x512 face
crops were visually inspected. Crowded foreground occlusion makes estimated leg
joints uncertain. Native replacement conditioning includes background and masks.
Pose extraction completed; VAE preprocessing took 45.42 seconds and the separate
reference encoder 1.60 seconds. MPS sampling loaded the correct architecture with no missing/unexpected weights,
but the resource supervisor stopped it before step one: system swap rose from
6.06 GB to a sampled peak of 11.50 GB in under a minute. Pressure reached warning
level 2. After exit, pressure returned to normal and swap fell. A single CPU-only
remedy is in progress; it uses the same physical memory and is not assumed to fit.

At this stage, known project storage including environments, caches, diagnostics
and the earlier Swift cache totals about 21 GB; important-usage available capacity
is about 91 GB. Fifteen custom tests and dependency consistency checks pass.
The pose preprocessor retains upstream licenses; its OpenPose-derived portions
carry non-commercial terms. These are separate from model moderation checks.

Private media, crops, annotations, raw diagnostics, intermediate tensors and
outputs remain excluded from Git. No media has been uploaded.

### CPU Animate benchmark in progress

The CPU-only retry completed step one in 219.46 seconds with normal pressure and
declining swap. The initial sampling estimate is 70–90 minutes for this diagnostic
segment; it continues under the same resource limits. This is not a quality result.

Earlier RSS logs sampled the direct child only; where sandbox-exec remains a
wrapper, those figures undercount inference descendants. They are not total model
or unified-memory measurements. The supervisor now records process-tree RSS and
CPU time where permitted, labeling direct-child-only fallback. Summed RSS can
double-count shared pages. System-wide pressure/swap/disk remain the stop criteria.

### First CPU Animate completion

The 13-frame, 192x256, 20-step Euler/CFG-5 CPU run completed sampling in
3856.97 seconds (64m17s). Across 383 resource samples, pressure stayed normal
(level 1), swap fell from 8.45 GB to 6.64 GB, and available disk never fell below
88.43 GB. Direct-child sampled RSS peaked at 6.04 GB, with the scope caveat above.
Decoding completed in 33.05 seconds. Representative output frames have distorted
faces and increasing colored artifacts; this is a quality failure. It is not a
finished replacement, despite successful CPU execution.

A subsequent source check found that the original Animate configuration defaults
to 20 steps, shift 5, guidance 1.0 and UniPC. CFG 5 in the first experiment was an
unverified choice, not that default. If quality is inadequate, the next bounded
comparison must correct these settings before attributing failure to quantization
or increasing model size. Source:
https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/configs/wan_animate_14B.py

The bounded follow-up keeps the same weights, source/reference controls and seed,
but uses upstream UniPC and CFG 1 on CPU. Expected sampling time is 30–40 minutes
based on the measured CFG-5 run and removal of the extra guidance pass. No larger
weights are downloaded before this correction is evaluated.

### Upstream-default comparison and VAE isolation

UniPC/CFG-1 CPU sampling completed in 2124.71 seconds (35m25s); decoding took
29.88 seconds. The same severe colored artifacts and inadequate likeness persist.
Across 211 samples there were no critical-pressure readings; pressure peaked at
warning level 2, swap fell from 6.63 GB to 5.56 GB, and process-tree sampled RSS
peaked at 5.23 GB. This configuration fails quality despite successful execution.

A source-video VAE encode/decode round trip completed in 47.24 seconds. Visual
inspection shows clean reconstruction without the generated neon/striping defects.
Per-frame normalized mean absolute pixel error ranges from 0.0074 to 0.0201.
This isolates the obvious corruption to generation rather than ordinary source
reconstruction; it does not prove the exact cause within generation.

One higher-precision comparison uses the pinned Q3_K_S file, 7,969,675,072 bytes,
with all other corrected settings and inputs fixed. This choice follows measured
CPU headroom of roughly 4–5 GB; its additional file size is not asserted to equal
peak memory growth. Download and inference retain the same resource guards.
Projected known project storage is about 29 GB. No output is accepted yet.

### Q3 result and shared-stage checks (2026-09-27)

Animate Q3_K_S sampling completed in 2099.43 seconds (34m59s), followed by
29.39 seconds of decoding. Across 209 resource samples, pressure remained normal
(level 1), swap fell from 5.512 GB to 5.454 GB, sampled process-tree RSS peaked at
5.382 GB, and available disk remained at least 83.577 GB. The same severe facial
distortion, striping and late-frame color artifacts persist. Increasing precision
did not produce an acceptable replacement. The diagnostic MP4 decodes completely:
640x360, 13 frames at 12.5 fps, with video and audio both starting at zero and
lasting 1.04 seconds. Representative frames were inspected; visual playback has
not been verified.

A CPU FP32 text-encoding diagnostic completed in 50.87 seconds. Both prompt
tensors are bit-for-bit equal to those from the prior FP16 text setting. This
rules out a difference caused by that flag in this configuration, not all possible
text-encoder defects. Memory pressure briefly reached warning, without swap growth.

A final pipeline comparison is running the already-downloaded original VACE
checkpoint on CPU, retaining the Euler test's inputs, masks, seed and settings.
This bypasses both GGUF diffusion loading and Metal without downloading more
weights. Results remain pending. It is not an accepted generation configuration.

Measured allocated storage for dependencies, environments, work, project caches
and the known external diagnostic cache totals about 29.30 GB. Model alternatives
remain counted; nothing has been removed to hide storage costs.

The low-resolution diagnostics are below the documented VACE 480p target. A
480x640, 13-frame version is prepared to test this remaining limitation after
the CPU comparison. Higher-resolution compositing has an explicit resize option
and a preservation test; generation at that size has not yet been attempted.
Source: https://github.com/ali-vilab/VACE/blob/main/README.md

### Original-checkpoint CPU result and 480p test

Original-checkpoint VACE CPU sampling completed in 664.15 seconds including
loading; decoding took 56.76 seconds. The 67 resource samples peaked at warning
pressure (2), swap increased from 5.454 GB to 6.846 GB, sampled RSS peaked at
5.602 GB, and available disk remained above 82.451 GB. Representative frames
closely resemble the earlier Q8/Metal Euler result: white T-shirt replacement,
but stylized/different facial features, pose drift and colored patches. This
fails quality and weakens the hypothesis that GGUF or Metal alone caused the
VACE defects. The diagnostic MP4 passes full FFmpeg decoding.

The next bounded test is the prepared 480x640, 13-frame CPU VACE job, using the
original checkpoint, the same source interval/masks and a higher-resolution crop
from the unchanged reference collage. Sampling is estimated at 40–70 minutes
from the 11-minute low-resolution CPU benchmark, increased spatial area and
reduced temporal count. This estimate is not a measured 480p benchmark. All
resource thresholds remain active. No more model downloads are needed.

The first untiled 480p preparation was stopped before sampling: swap rose from
6.830 GB to a sampled peak of 13.076 GB, with warning pressure and the >2 GB/min
swap-growth threshold exceeded. The child exited with failure; no prepared
conditioning was accepted. Pressure recovered to normal after termination.

A bounded remedy uses native ComfyUI VAE tiling, not a replacement engine:
encoding tiles are 256x256 pixels, 9 frames, with 64-pixel/5-frame overlap;
decoding tiles are 32x32 latent pixels, 3 latent frames, with 8-pixel/1-frame
overlap. A 480p source round trip is being checked for dimensions, reconstruction
and resource use before retrying preparation. Stop thresholds are unchanged.

The temporal/spatial tiled round trip completed in 507.73 seconds with normal
pressure throughout. It produced all 13 expected 480x640 frames, but representative
frames show temporal ghosting and a motion discontinuity; normalized per-frame
MAE ranges from 0.00287 to 0.06072. This result was rejected before generation.
A corrected native spatial-only tiling check retains all 13 frames in each tile,
removing temporal tile boundaries. Spatial tiles and resource thresholds remain
unchanged. The rejected round-trip artifacts are retained privately for comparison.

Spatial-only VAE tiling completed the source round trip in 470.86 seconds.
All 13 expected 480x640 frames were produced. Representative frames no longer
show the temporal ghosting/discontinuity; normalized per-frame MAE is
0.00287–0.00828. Pressure stayed normal (1), swap fell from 7.364 GB to 7.238 GB,
and sampled RSS peaked at 5.352 GB. This verifies the bounded VAE memory remedy
for this short sequence. Preparation is now using spatial-only tiles (256x256
pixels with 64-pixel overlap; decode 32x32 latent pixels with 8-pixel overlap),
retaining the full temporal window. It does not establish generation quality.

480p preparation with spatial-only tiling completed in 364.97 seconds. Across
37 samples, pressure stayed normal, swap fell from 7.213 GB to 7.045 GB, sampled
RSS peaked at 4.727 GB, and available disk remained above 83.729 GB. Sampling
is now running with the original checkpoint on CPU.

The first three 480p sampling steps take about 225–232 seconds each, revising
the initial estimate to roughly 75–85 minutes for sampling. Early memory pressure
is warning (2), with stable swap below the growth stop threshold. No decoded
480p generation is available yet.

The 13 prepared source crops match the corresponding compositing source frames
bit-for-bit, ruling out a frame-selection mismatch for this diagnostic. Separate
post-compositing masks refine foreground-person occlusions in the last frames;
the original inference masks remain unchanged so the conditioning is traceable.
These mask refinements are manually reviewed, not a demonstrated automatic
segmentation pipeline for the full clip.

480p original-checkpoint CPU sampling completed in 4641.95 seconds (77.4 minutes).
Across 461 resource samples, pressure peaked at warning (2), swap rose from
6.986 GB to 8.501 GB, sampled RSS peaked at 5.135 GB, and available disk stayed
above 80.298 GB. Spatial decode completed in 330.85 seconds. The 13-frame
640x360 composited MP4 has matching 1.040-second video/audio durations and zero
start offsets, and passes complete FFmpeg decoding.

Representative frames show a substantial quality improvement: recognizable
reference appearance, white T-shirt, dark trousers, and replaced body/silhouette,
without the earlier neon corruption. However, the original arm-down pose becomes
hands in pockets and the expression follows the smiling reference too closely.
This is a saved diagnostic preview, not an accepted faithful final edit.

A bounded 480x640 Animate Q3 CPU test uses the same interval and reference,
fresh pose/face controls, native spatial VAE tiling, official 20-step/CFG1/shift5
settings and UniPC. Estimated sampling is 3–5 hours from the smaller benchmark;
this remains an estimate until measured. No larger generation weights are needed.

The 480p VACE preview also reaches its expected end time in the local browser
player without a media error. Its re-encoded audio has zero-lag correlation
0.999022 with the corresponding source interval (16,640 mono samples at 16 kHz).
Animate's 480p preparation completed in 382.16 seconds; sampling is running.

SAM2 tiny preprocessing source/weights have been pinned and downloaded for
full-clip occlusion masks. Its CPU runner is staged but has not yet been used for
inference. Prompt validation brings the local suite to 23 passing tests; pip
reports no broken requirements. Project files occupy about 29.6 GB, plus about
0.13 GB of known external caches, before any additional full-clip generation.
