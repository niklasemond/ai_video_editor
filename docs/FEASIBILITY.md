# Local character replacement: feasibility assessment

Assessment date: 2026-09-26. **Decision: do not install or download model weights on the specified 16GB Mac using the inspected implementation.** This is an evidence-backed stop before installation, not a demonstrated replacement workflow and not a proof that every possible implementation is infeasible.

## Baseline and evidence boundaries

- Target repository baseline: `4f21b5db45cc5f45d755facb22cbe00c9a5b1a3a`. Its complete tree contains only `README.md` (a title). No application, tests, CI, dependency manifest, or repository-level AGENTS instructions exist at that revision.
- Spielberg revision: [`f692f93d73af996169244e9d8fa0178d6383f0d6`](https://github.com/dtellz/spielberg/tree/f692f93d73af996169244e9d8fa0178d6383f0d6). Complete recursive tree: 47 files, no tests or CI. Inspected its packaging, converter, model loader, pipeline, preprocessing, server, browser code, license and NOTICE; scanned 39 code/configuration files for remote calls and filtering indicators.
- Hardware specification is **user supplied**: M1, 16GB unified memory, Sonoma 14.5, approximately 110GB available. Actual free disk, available memory, swap, OS version and installed prerequisites **could not be measured**. The local command runner failed before execution because of an unsupported symlinked writable root; an independent local Node runtime also failed to start. No personal paths or raw diagnostics are included here.
- GitHub access is authenticated. This documentation is committed through the GitHub connector; there is no successful local clone or local commit claim. Local parent-directory instructions could not be read.
- No package installs, weight downloads, conversions, preprocessing, inference, or media uploads were performed. No upstream code was copied or modified.

## Blocking source findings

### Missing required engine modules

The [complete Git tree](https://api.github.com/repos/dtellz/spielberg/git/trees/f692f93d73af996169244e9d8fa0178d6383f0d6?recursive=1) has no `engine/vendor/mlx_video/models/` directory. Nine import statements reference seven missing modules under `mlx_video.models.wan_2`: `config`, `attention`, `rope`, `wan_2`, `utils`, `scheduler`, and `convert`.

For example, [config.py line 20](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/animate/config.py#L20) imports `WanModelConfig` from the missing tree. [engine/__init__.py](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/__init__.py) prioritizes the incomplete vendor package. There is neither a submodule declaration nor a declared external mlx-video dependency that supplies these files. This is a reproducibility defect visible without executing the code; an actual import failure was not run.

### Conversion expands the full model before quantization

[convert.py](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/weights/convert.py) loads GGUF into dequantized f16/f32 tensors, optionally merges relighting weights, loads the full model, and explicitly evaluates its parameters **before** applying 4-bit quantization (group size 64; selected transformer linear layers only). It is not a streaming conversion.

Approximately 17 billion parameters at two bytes each means roughly **34GB of unquantized weights**, excluding conversion buffers, f32 tensors, LoRA intermediates, the OS and other processes. This is a source-derived estimate, not measured resident memory. Reducing frames, resolution or steps does not reduce this setup requirement. A deliberate swap-heavy trial would conflict with the requirement to avoid sustained severe memory pressure.

The auxiliary converter also describes an approximately 11GB bf16 T5 encoder that its loader upcasts to f32 (roughly 22GB). The referenced loader is missing, so that behavior is an upstream comment, not independently verified implementation evidence.

### Other issues to resolve before a future test

- [Pose preprocessing](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/preprocess/wan_pose/pose2d.py) imports torch at runtime; the renderer imports matplotlib. Torch is only declared in the conversion extra, and matplotlib is not declared. The selected pose path uses CPU ONNX execution, not CoreML.
- [Pipeline](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/animate/pipeline.py) silently substitutes the base DiT if replacement relighting weights are absent, and zero CLIP features if the converted visual encoder is absent. Both must become actionable failures for this workflow.
- [Replacement masks](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/preprocess/replace.py) default to a dilated convex hull of pose points. Hair, clothing outlines, occlusions and nearby people therefore need visual validation. The optional SAM2 integration is untested; its API and device selection need checking.
- [Runner](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/server/runner.py) encodes generated frames but does not remux original audio.
- [Server](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/server/app.py) and [queue](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/server/jobs.py) provide serial jobs and status, but no cancellation endpoint, resource watchdog, clip start selection, or bounded parameter validation. No browser service was launched.

## Resource budget

Decimal GB throughout; these are planning estimates, not local disk measurements.

The [published original files](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/tree/main), [GGUF listing](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/tree/main), and preprocessing listings give:

| Download | Approximate GB |
| --- | ---: |
| Q4_K_M DiT GGUF | 11.5 |
| Original T5 encoder | 11.4 |
| Original CLIP | 4.77 |
| VAE | 0.508 |
| Relighting checkpoint used by converter | 2.87 |
| Additional relighting adapter downloaded by the default glob | 2.87 |
| YOLO detector | 0.0617 |
| ViTPose | 2.55 |
| **Subtotal before tokenizers/configs** | **36.53** |

The downloader fetches both `relighting_lora.ckpt` and `relighting_lora/**`; only the checkpoint is consumed by this converter. Converted T5 (~11.4GB), CLIP visual (~2.5GB estimate), VAE (~0.51GB), and base plus relighting DiTs (~9–11GB each) bring the standard route to approximately **69–73GB before environments, caches and outputs**. This conflicts with the 60GB total-project ceiling.

Skipping the unused LoRA format and base DiT would reduce the estimate to approximately **57–59GB**, still before environments, caches, temporary downloads and outputs. That is not a verified compliant plan. It also does not solve the conversion-memory blocker. Optional SAM2 would add further code and weights.

The separate 30GB-free requirement remains unverified because actual free space is unavailable. Do not treat the user's approximate 110GB as a measurement. Downloads must remain disabled until both constraints can be checked, including external caches and conversion intermediates.

## Filters, remote dependencies and model restrictions

The inspected application code contains no observed prompt blacklist, NSFW detector, moderation API, paid inference API, or automatic cloud inference fallback. This is a bounded source review, not a guarantee about every transitive dependency or learned model behavior. No filters or moderation services were added.

Setup depends on GitHub, Python/npm package distribution, and Hugging Face downloads. [models.py](https://github.com/dtellz/spielberg/blob/f692f93d73af996169244e9d8fa0178d6383f0d6/engine/weights/models.py) does not pin model revisions. During generation, `AutoTokenizer.from_pretrained("google/umt5-xxl")` is called without a pinned revision or `local_files_only=True`. Vendored download utilities can also fetch uncached models. The downloaded nested tokenizer directory is not explicitly selected by this call. Offline operation must use pinned local resources, fail on missing assets and undergo a process-isolated offline generation test.

Spielberg original code is MIT; its NOTICE attributes the mlx-video base to Prince Canuma (MIT) and the Wan port to Alibaba (Apache-2.0). The [Wan model card](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/blob/main/README.md) identifies Apache-2.0 and separately states behavioral restrictions concerning unlawful or harmful use, harmful disclosure of personal information, misinformation and targeting vulnerable populations. Its linked LICENSE.txt returned 404 during this review; do not imply the complete model license package was verified. Preserve all applicable notices when code or weights are eventually incorporated.

These stated terms differ from executable filtering. Training and model capacity can still limit identity, anatomy, clothing, motion, background fidelity and temporal consistency. Neither unrestricted output nor faithful replacement is promised.

## Compatibility and reproducibility record

[MLX installation documentation](https://ml-explore.github.io/mlx/build/html/install.html) lists Apple Silicon, native Python >=3.10 and macOS >=14.0. The supplied M1/Sonoma specification meets that broad platform description; exact package, Metal operation and memory compatibility remain untested.

Spielberg requires Python >=3.11 and mlx >=0.22.0, but its Python dependencies are not locked. No compatible runtime versions were installed or verified here. The frontend has a package lock, but no build was run.

Expected inputs to a future replacement conversion are the files in the budget table, the local T5 tokenizer and config. Upstream conversion settings are Q4_K_M source, native 4-bit selected linear layers, group size 64, relighting merge scale 1.0; auxiliary T5 bf16, VAE f32 and CLIP visual f32.

One available integrity identifier was inspected: [Q4_K_M GGUF SHA256](https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF/blob/main/Wan2.2-Animate-14B-Q4_K_M.gguf) `43d720f243c3cdf5346ca05b525ec662f89bc5ebeb8e998dc347b840406cdfa6`. It was not validated against a downloaded file. Exact weight revisions and remaining hashes are not locked; no deployable model manifest is claimed.

## One alternative assessed

[xocialize/scail-2-mlx at 9ca63749ff9ff261451fc2681c1bd84799950d2a](https://github.com/xocialize/scail-2-mlx/tree/9ca63749ff9ff261451fc2681c1bd84799950d2a) implements the same general reference-image/driving-video character replacement task. Its README, port plan, pipeline and quantization recipe were inspected.

The author reports ~34GB active/~47GB peak memory for an M5 Max animation configuration, leaves replacement validation and smaller-Mac memory handling open, and reports q4 fidelity problems. Its quantizer loads/evaluates the bf16 model before quantizing. These are upstream reports plus source findings, not local benchmarks. It does not establish a safe 16GB replacement route. It was not installed; no second alternative was investigated.

## Test status and presets

| Check | Result |
| --- | --- |
| Target baseline and remote repository instructions | Inspected; README only |
| Complete upstream tree vs required Wan imports | Blocker confirmed by static comparison: nine references, seven absent modules |
| Dependency, conversion, network and replacement-path review | Completed at pinned code revision |
| Actual hardware, free disk, prerequisites, local instructions | Blocked by local execution infrastructure |
| Existing target tests/lint | Not applicable: none exist at baseline |
| Upstream tests | None present in the inspected tree; no runtime checks executed |
| Imports, Metal, media decode, pose/masks, model loading | Not run |
| End-to-end generation, visual quality, timing, peak memory | Not run; no generated output |
| Audio sync, offline generation, cancellation/recovery | Not run |
| Resource/error-path simulations | Not run; no custom runtime introduced |

No Quick test or Quality preset is released. The implementation exposes area-based dimensions rounded to multiples of 16, 4n+1 clip lengths, steps, seed and solver. Those controls alone do not validate output or make the full-weight conversion fit. Quality must remain unavailable until a real output is reviewed. No speed estimate is defensible without a successful measured run.

## What must change to resume

1. Restore local command execution in a supported workspace and read applicable local instructions; measure disk, RAM, memory pressure, swap and prerequisites before any installation.
2. Obtain the missing engine files at a compatible, attributable revision and fix undeclared dependencies. Do not improvise a substantial model port.
3. Establish a memory-feasible route: complete compatible preconverted quantized components (including relighting and a memory-feasible T5 encoder), or an explicitly approved conversion/inference redesign. A higher-memory Mac could make the current conversion more plausible, but no minimum working hardware configuration was verified here.
4. Build an exact component manifest and peak-disk schedule under 60GB, preserving 30GB free. Pin packages, revisions and checksums; localize caches and prevent duplicate downloads.
5. Only then perform a monitored minimal replacement run, ask for the user's source clip and reference image, and validate actual output. Add strict required-component checks, audio remux, localhost binding, cancellation and recovery before offering a usable interface.
6. Compare at most a small justified set of configurations. Publish both presets only after measured runs and visual inspection; estimate any substantially longer run from those measurements.

This assessment intentionally stops before unsafe or non-reproducible installation. The requested complete workflow remains unavailable.
