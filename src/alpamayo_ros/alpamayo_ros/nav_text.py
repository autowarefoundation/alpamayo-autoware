# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""Turn an Autoware lanelet2 map plus a mission route into a navigation instruction.

Extracted verbatim from the Alpamayo 1.5 node so the Alpamayo 2 Super navigation-CFG
path can reuse it without standing up a ROS node. The wording produced here
("Turn left in 30m" / "Continue straight") is what the models were conditioned on, so it
should not be paraphrased freely.

The lanelet2 Python bindings live in an Autoware workspace, not in the base ROS install,
so importing them is optional: callers get ``None`` and can disable navigation instead of
crashing.
"""

from __future__ import annotations

import numpy as np

try:
    import lanelet2
    from autoware_lanelet2_extension_python.projection import MGRSProjector

    HAS_LANELET2 = True
except ImportError:  # pragma: no cover - depends on the sourced Autoware workspace
    HAS_LANELET2 = False

#: Lanelet subtypes the ego can drive on. Anything else (crosswalks, parking) is ignored.
DRIVABLE_SUBTYPES = ("road", "highway", "road_shoulder", "bicycle_lane")


def load_lanelet_map(map_path: str) -> dict[int, dict] | None:
    """Load a lanelet2 map and keep only what navigation needs.

    Args:
        map_path: Path to ``lanelet2_map.osm``.

    Returns:
        ``{lanelet_id: {"centerline", "turn_direction", "center"}}``, or ``None`` when the
        lanelet2 bindings are unavailable.
    """
    if not HAS_LANELET2:
        return None

    projection = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
    ll2_map = lanelet2.io.load(map_path, projection)

    lanelet_info: dict[int, dict] = {}
    for lanelet in ll2_map.laneletLayer:
        attributes = lanelet.attributes
        subtype = attributes["subtype"] if "subtype" in attributes else ""
        if subtype not in DRIVABLE_SUBTYPES:
            continue
        centerline = np.array([(p.x, p.y, p.z) for p in lanelet.centerline])
        lanelet_info[lanelet.id] = {
            "centerline": centerline,
            "turn_direction": (
                attributes["turn_direction"] if "turn_direction" in attributes else ""
            ),
            "center": np.mean(centerline[:, :2], axis=0),
        }
    return lanelet_info


def compute_nav_text(
    lanelet_map: dict[int, dict] | None,
    route_lanelet_ids: list[int],
    ego_pos_map: np.ndarray,
) -> str | None:
    """Derive a navigation instruction for the ego's position along the route.

    Finds the closest route lanelet by its first centerline point. If the ego already sits
    on a turning lanelet the turn is reported immediately; otherwise the next turn on the
    route is reported with the accumulated distance to it.

    Args:
        lanelet_map: Output of :func:`load_lanelet_map`.
        route_lanelet_ids: Preferred primitive IDs in route order.
        ego_pos_map: Ego position in the map frame, ``(>=2,)``.

    Returns:
        e.g. ``"Turn left"``, ``"Turn right in 42m"``, ``"Continue straight"``, or ``None``
        when no map or route is available.
    """
    if lanelet_map is None or not route_lanelet_ids:
        return None

    ego_xy = ego_pos_map[:2]

    best_idx = 0
    best_dist = float("inf")
    for i, lanelet_id in enumerate(route_lanelet_ids):
        info = lanelet_map.get(lanelet_id)
        if info is None:
            continue
        dist = np.linalg.norm(info["centerline"][0, :2] - ego_xy)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    current_info = lanelet_map.get(route_lanelet_ids[best_idx])
    if current_info is not None and current_info["turn_direction"] in ("left", "right"):
        return f"Turn {current_info['turn_direction']}"

    cumulative_dist = 0.0
    prev_pt = ego_xy
    for lanelet_id in route_lanelet_ids[best_idx + 1 :]:
        info = lanelet_map.get(lanelet_id)
        if info is None:
            continue
        first_pt = info["centerline"][0, :2]
        cumulative_dist += np.linalg.norm(first_pt - prev_pt)
        prev_pt = first_pt
        if info["turn_direction"] in ("left", "right"):
            return f"Turn {info['turn_direction']} in {int(round(cumulative_dist))}m"

    return "Continue straight"
