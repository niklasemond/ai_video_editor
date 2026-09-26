# Reproduce the experimental local test

This is a development harness. A visually acceptable replacement and browser
interface have not yet been demonstrated. Do not treat the first diagnostic
video as a completed edit.

Requirements: Apple Silicon Mac, Python 3.12, FFmpeg and Git. The measured test
host uses macOS 14.5 and 16 GiB unified memory. No system Python packages are
modified. Run from the repository root in a normal terminal with local GPU access.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.lock.txt
mkdir -p .local
git clone https://github.com/Comfy-Org/ComfyUI .local/ComfyUI
git -C .local/ComfyUI checkout b8730510db30c8858e1e5d8e126ef19eac395560
git clone https://github.com/city96/ComfyUI-GGUF .local/ComfyUI-GGUF
git -C .local/ComfyUI-GGUF checkout 6ea2651e7df66d7585f6ffee804b20e92fb38b8a
.venv/bin/python scripts/download_models.py
.venv/bin/python -m unittest discover -s tests -v
```

Downloads use individual immutable-revision URLs, HTTPS certificate verification,
resumable `.part` files and published SHA-256 checksums. Failed or corrupt files
are not loaded. The complete FP16 checkpoint recovers and validates the convolution omitted
from the community GGUF. It can also be selected for an original-checkpoint CPU
diagnostic with `model: "wan2.1_vace_1.3B_fp16.safetensors"` and `device: "cpu"`. See LOCAL_TEST.md for the exact defect and comparison.

A private job directory under `work/` contains `frames/000.png` onward,
`masks/000.png` onward (white means replace), a cropped and aspect-padded
`reference.png`, and `config.json`. Masks must follow the entire visible character
through the segment and preserve foreground occluders; no single static mask is
assumed valid. The harness currently needs reviewed, externally prepared masks.

Example configuration fields (no input media is distributed):

```json
{
  "width": 192, "height": 256, "frames": 25,
  "fps": 25, "start": 0, "crop": [0, 0, 192, 256],
  "steps": 20, "cfg": 6, "shift": 5, "seed": 20260926,
  "blank_masked_source": true,
  "positive": "Describe the reference character and source performance.",
  "negative": "Describe visible defects to avoid."
}
```

All frames, masks and the padded reference must match the generation dimensions.
Frame count is 4n+1. `crop` uses source-video coordinates. For this harness the
generated dimensions must equal the crop dimensions unless `resize_to_crop` is
explicitly true. That option resizes the generated crop and its mask into the
source crop before compositing; it does not enlarge the full source frame.
Full-clip segmentation is not implemented.

Execute each stage sequentially. Every stage exits to release its models before
the next one starts. Substitute the prepared private job directory for `work/job`.

```sh
.venv/bin/python scripts/resource_watch.py work/job/text-resources.jsonl /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -u scripts/vace_stage.py text work/job
.venv/bin/python scripts/resource_watch.py work/job/prepare-resources.jsonl /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -u scripts/vace_stage.py prepare work/job
.venv/bin/python scripts/resource_watch.py work/job/sample-resources.jsonl /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -u scripts/vace_stage.py sample work/job
.venv/bin/python scripts/resource_watch.py work/job/decode-resources.jsonl /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' .venv/bin/python -u scripts/vace_stage.py decode work/job
.venv/bin/python scripts/assemble_preview.py work/job INPUT_VIDEO.mp4
```

Stop the foreground supervisor with Ctrl-C to terminate its child process group.
Successful stage intermediates are retained. A failed stage must be investigated
before retrying; earlier completed stages need not be rerun. `.tmp` stage files
are not accepted as completed results. The supervisor does not set an arbitrary
runtime limit. It stops on resource limits described in LOCAL_TEST.md.

The network-denial profile applies only to the inference child; the supervisor
can query macOS capacity APIs. Inference has no cloud fallback or moderation API.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is a local CPU fallback for unsupported Metal ops.
It does not relax macOS memory protections. No NVIDIA-specific kernels are used.

Inspect the generated sequence and composited preview for identity, clothing,
silhouette, motion, occlusion and flicker. Successful execution is insufficient.
No Quality preset is claimed until a visually acceptable configuration is measured.

## Experimental Animate stage extension

Clone `Fannovel16/comfyui_controlnet_aux` into `.local/controlnet_aux` and
check out the revision in `upstream.lock.json`. Its per-component licenses apply.
The pinned requirements include CPU ONNX Runtime and pose drawing dependencies.
Set `engine` to `animate`, `sampler` to `uni_pc`, `cfg` to `1`, and
`device` to `cpu` in the private config. These match the verified original
Animate sampler/guidance defaults; the earlier Euler/CFG-5 experiment failed
visually. Prepare
source frames, per-frame masks and reference as above, then run `pose_controls.py`
under the same monitored network-denial wrapper. Inspect `poses/` and `faces/`
before running `prepare`. Run the additional `vision` stage in a separate process
before `sample`. Reuse the text encoder and VAE; each required file is verified
before loading. This remains a feasibility experiment, not a validated preset.

For the experimental 480p job, `tiled_vae: true` uses native spatial tiles while
retaining the full short temporal window. Temporal tiling was rejected after an
actual source round trip showed ghosting. A 13-frame spatial-only round trip
passed visual/resource checks; see LOCAL_TEST.md. This is not evidence that long
unsegmented videos fit memory. Use the `roundtrip` stage to inspect reconstruction
before relying on a changed VAE configuration.
