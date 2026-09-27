# Reproduce the experimental local test

This is a development harness. A visually acceptable replacement and browser
interface have not yet been demonstrated. Do not treat the first diagnostic
video as a completed edit.

Use a workspace and private output folder outside cloud-synced locations.
Git ignore rules and the inference network sandbox do not stop a separate sync
client from uploading files. The media stages reject recognized Dropbox paths
unless an ancestor has an explicit ignore attribute; this is not a universal
backup/sync audit. For a Dropbox workspace, exclude task-created private data
folders before creating assets, following the applicable Dropbox version's
instructions: https://help.dropbox.com/sync/ignored-files . Ignoring an already
synced folder keeps local files but removes remote/other-device copies. Never
apply that change to unrelated source folders without authorization.

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

Set `composite_masks: true` only when a reviewed `composite-masks/` sequence is
present. It refines final compositing separately from the original `masks/`
sequence used by inference, for example to retain foreground occluders. It must
have the same frame count and generation dimensions. Original conditioning masks
remain available for reproduction and inspection.

## Experimental tracked-mask preprocessing

The pinned SAM2 tiny source belongs in `.local/SAM2`; its 156 MB checkpoint is
listed in `models.lock.json`. The runner imports source directly, so no CUDA
extension or SAM2 package build is needed. Dependencies are pinned in the lockfile.
The CPU runner accepts contiguous, zero-based JPEG frames, a JSON list of
`{frame, points: [[x,y], ...], labels: [1,0,...]}` records, and a new output folder:

```sh
.venv/bin/python scripts/resource_watch.py work/mask-resources.jsonl \
  /usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  .venv/bin/python -u scripts/segment_person.py \
  work/source-jpeg work/person-prompts.json work/person-masks
```

Use positive points on the performer and negative points on foreground people,
including a first-frame prompt. Review every resulting mask sequence before use.
An `INCOMPLETE` marker remains after interrupted/failed tracking; use a new output
folder for recovery. The tracker never overwrites an earlier result. CUDA-only
hole-filling postprocessing is disabled. SAM2 code/checkpoints use Apache-2.0;
upstream licenses remain in its checkout. No application moderation service or
content classifier was found in the inspected loader/tracker path. This is a
bounded inspection, not a claim about every dependency or unrestricted capability.

Prompt-validation tests pass; actual tracked-mask inference is still pending.

## Source-aligned job preparation

`prepare_job.py` accepts a local video, reference image, reviewed full-frame masks,
settings JSON and a new output directory. Masks use the source's zero-based
decoded frame indices (`00000.png`, etc.). An optional separate compositing-mask
folder protects reviewed foreground occlusions.

```sh
.venv/bin/python scripts/prepare_job.py \
  --source input.mp4 --reference reference.png --masks work/person-masks \
  --settings work/settings.json --output work/job
```

Settings contain the usual stage configuration, a source-pixel `crop` rectangle,
`start`, `fps`, and a bounded 4n+1 frame count (5–25). An optional `reference_crop`
rectangle performs ordinary cropping before aspect-preserving padding; it does
not synthesize or retouch the reference. `mask_dilate` defaults to zero and can
expand inference masks by up to 16 source pixels, after visual review. Separate
compositing masks are not dilated automatically.

Source timestamps select each frame once. The saved full-size `source-frames/`
sequence is reused by compositing, and a source-file SHA256 check prevents
accidentally combining prepared frames with a different video's audio. A partial
preparation keeps an `INCOMPLETE` marker and is refused by generation/compositing.
Recover into a fresh output directory. This helper prepares media only; model
stages and visual acceptance are still required.

For a following Animate segment, `--continue-from work/previous-job` copies the
last five raw generated frames into the new job's native `continue_motion` input.
The new segment must begin at those same source frame indices. Source hash,
engine/model, frame rate, crop, dimensions and reference pixels must match.
`--continue-frames 1` is also supported by the native node. Missing, corrupt or
misaligned continuation inputs are rejected before model loading. This wires
upstream functionality; actual continuation quality remains unverified. Individual
segment previews include the overlap, which must not be duplicated when joining.

At the end of a clip only, `pad_last: true` permits up to three repeated final
source frames to meet the model's 4n+1 requirement. The padded inputs are recorded
and preview export is capped at the remaining source duration. Padding cannot
be used to fabricate a long continuation past the end of the source.

The generation crop must enclose every selected performer mask, including
requested dilation. Preparation refuses a crop that would silently omit visible
parts of the character. Widen the crop or correct a mistaken mask before retrying.
