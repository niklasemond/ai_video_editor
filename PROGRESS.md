# Progress

Last updated: 2026-09-26.

**Status: VACE and Q2 Animate tests run locally but fail visual quality; one Q3 comparison is in progress.**
See [the local test record](docs/LOCAL_TEST.md) for current measurements and fixes.
The Spielberg findings below are historical; that installation is not being resumed.

## Historical assessment (before local execution was restored)

The statements below describe the earlier assessment only. Current runtime evidence
and implementation status are in docs/LOCAL_TEST.md.

### Completed

- Authenticated GitHub access; inspected the README-only target baseline.
- Pinned and reviewed Spielberg at `f692f93d73af996169244e9d8fa0178d6383f0d6`.
- Compared its complete file tree with imports: seven required Wan modules are absent.
- Reviewed conversion memory, disk budgeting, dependencies, replacement handling, remote calls, filtering indicators and attribution.
- Assessed exactly one alternative, SCAIL-2 MLX, at `9ca63749ff9ff261451fc2681c1bd84799950d2a`; no verified 16GB route found.
- Published the assessment on `codex/local-character-replacement-assessment` through the GitHub connector.
- Added exclusions for private media, model files, environments, caches, outputs, logs and local secret-bearing configuration. Git ignore patterns do not prevent intentional force-add; inspect staged content before any future commit.

## Measurements and checks

- No on-Mac performance or resource measurements are available. The command runner failed before execution, and the independent local Node runtime failed to start.
- No existing target tests, lint configuration or application code were present at baseline.
- Source inspection and remote tree/import comparison completed; these do not verify inference.
- No installation, model download, conversion, server launch or generation occurred.
- No user assets were requested or uploaded, because the earlier feasibility gate failed.
- Quick test and Quality presets are unavailable, not silently substituted or marked validated.
- Runtime error cases, cancellation, offline operation, visual replacement and original audio preservation remain untested.

## Blockers

1. Published Spielberg tree is incomplete.
2. Converter evaluates full unquantized weights before 4-bit quantization; estimated weight memory alone is approximately 34GB.
3. Standard retained downloads plus converted components are estimated at 69–73GB before environments/caches/outputs, exceeding the 60GB ceiling.
4. Actual disk and hardware availability cannot yet be checked locally.
5. Upstream lacks required audio preservation, cancellation and fail-fast required-component behavior.

## Next steps

Restore local execution and measure resources before any install. Obtain a complete compatible upstream engine and a verified memory-feasible component/conversion route. Recompute the full peak disk budget and pin dependencies and models. Only then request the source clip/reference image and run the monitored replacement test. A substantial port, inference rewrite or further alternative search requires discussion.

The branch contains documentation and ignore rules only. There are no upstream code modifications, copied licenses to maintain, generated media or sensitive logs in this change. Keep the pull request in draft and do not merge.
