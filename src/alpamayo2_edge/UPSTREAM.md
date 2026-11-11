# Vendored `alpamayo2_edge`

This directory is a vendored copy of NVIDIA's Alpamayo 2 Edge inference package.

| | |
|---|---|
| Upstream | https://github.com/NVlabs/alpamayo2-edge |
| Upstream path | `src/alpamayo2_edge` |
| Upstream commit | `02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e` ("Support the Alpamayo 2 Edge checkpoint built on Cosmos3-Edge (#8)", 2026-11-10) |
| Model weights | https://huggingface.co/nvidia/Alpamayo2-Edge, revision `f8951d095de3589b3face1dec3f4394c8ecedb38` (the one upstream's README pins; ~4.9 GB, bf16) |
| Code license | Apache-2.0 (per-file `SPDX-FileCopyrightText: NVIDIA` headers preserved) |

Vendored for the same reason `src/alpamayo1_5` and `src/alpamayo2_super` are:
`src/alpamayo_ros` imports the model package as a plain top-level module, and a colcon
workspace has no clean way to consume an external `uv` path dependency.

## Deviations from upstream

None in the files: every file here is byte-identical to upstream at the commit above.
`load_physical_aiavdataset.py` is kept, because it imports `physical_ai_av` only inside
its functions, so the package imports without the dataset SDK.

**Python 3.10 instead of 3.12.** Upstream's `pyproject.toml` declares
`requires-python = ">=3.12,<3.13"`. ROS 2 Humble ships Python 3.10, and the one newer
feature the package uses is the `z` format flag (3.11) in
`chat_template.relative_time_labels`. All modules parse under 3.10. The ROS node
(`alpamayo_ros/alpamayo2_edge_node.py`) replaces that one function at import time with an
equivalent that does not use the flag, leaving these files unmodified. Run under 3.10 this
way and under upstream's own 3.12 environment, upstream's navigation example (clip
`ea7bbd31-b7a5-4972-8dbd-7089e6b53de4`, t0 4.0 s, "Turn right in 30m", seed 43) gave
identical prompt tokens, generated tokens and trajectories.

Upstream pins `scipy>=1.16.0` and `numpy>=2.0`; scipy 1.16 and numpy 2.3 need Python 3.11,
so `requirements-alpamayo2-edge.txt` caps them at the last releases for 3.10. The other
pins (torch 2.8.0, torchvision 0.23.0, transformers 5.17.0) are upstream's.

## Re-vendoring

```bash
rm -rf src/alpamayo2_edge
cp -r <alpamayo2-edge-clone>/src/alpamayo2_edge src/alpamayo2_edge
find src/alpamayo2_edge -name __pycache__ -type d -prune -exec rm -rf {} +
```
Then restore this file, update the commit and revision above, and re-check that the
package still needs nothing from Python 3.11+ but the `z` flag.
