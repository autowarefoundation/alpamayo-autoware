# Vendored `alpamayo2_super`

This directory is a vendored copy of NVIDIA's Alpamayo 2 Super inference package.

| | |
|---|---|
| Upstream | https://github.com/NVlabs/alpamayo2 |
| Upstream path | `src/alpamayo2_super` |
| Upstream commit | `9596749e33552468f3ae70da8c237bdd0401e59a` ("Update README.md", 2026-08-03) |
| Model weights | https://huggingface.co/nvidia/Alpamayo2-Super (~71.6 GB, bf16, 32 files) |
| Code license | Apache-2.0 (per-file `SPDX-FileCopyrightText: NVIDIA` headers preserved) |

Vendored for the same reason `src/alpamayo1_5` is: `src/alpamayo_ros` imports the model
package as a plain top-level module, and a colcon workspace has no clean way to consume an
external `uv` path dependency.

## Deviations from upstream

- **`load_physical_aiavdataset.py` removed.** It is the only module that imports
  `physical_ai_av` (the PhysicalAI-AV dataset reader). The ROS node builds its model input
  from live camera/odometry topics instead, so the dataset dependency is not needed here.
  `inference_smoke.py` still references it, but only inside a function body, so importing
  the package does not fail. Running `python -m alpamayo2_super.inference_smoke` in this
  checkout will not work — that is expected.

- **Python 3.10 instead of 3.12.** Upstream's `pyproject.toml` declares
  `requires-python == "==3.12.*"`, but nothing in the source actually requires 3.11+: all
  35 modules parse under 3.10 and none use `Self`, `tomllib`, `except*`, `StrEnum`,
  `itertools.batched`, or `datetime.UTC`. ROS 2 Humble ships Python 3.10, so the package is
  imported unmodified under 3.10 here. Re-check this when re-vendoring a newer upstream.

No source files were otherwise modified.

## Re-vendoring

```bash
cp -r <alpamayo2-clone>/src/alpamayo2_super src/alpamayo2_super
rm -f src/alpamayo2_super/load_physical_aiavdataset.py
find src/alpamayo2_super -name __pycache__ -type d -prune -exec rm -rf {} +
```
Then update the commit hash above and re-run the Python 3.10 compatibility check.
