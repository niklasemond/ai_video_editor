# Local VACE test — in progress

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
  values are in the lock files. All three downloaded model hashes verified.
  VACE GGUF reports architecture `wan`, 1263 tensors and 438 VACE tensors, but
  subsequent full comparison exposed a missing required tensor (see below).
- Dependency consistency check passed. Thirteen custom unit tests passed, covering
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
78.01 seconds; sampling and visual evaluation are pending. No full-clip run or
browser interface is justified by the first result.

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
and character masks. It shares the UMT5/VAE stages, so the text-stage failure
must be resolved first. No 14B memory-fit claim or download is made yet.

Private media, crops, annotations, raw diagnostics, intermediate tensors and
outputs remain excluded from Git. No media has been uploaded.
