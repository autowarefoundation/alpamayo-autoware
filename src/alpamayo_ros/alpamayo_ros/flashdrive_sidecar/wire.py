# SPDX-License-Identifier: Apache-2.0
"""Dependency-light wire format for the FlashDrive sidecar IPC.

The ROS node (Python 3.10, ROS Humble venv) and the FlashDrive sidecar
(Python 3.12, torch 2.9.1) run in different interpreters with different torch
versions, so we must NOT rely on ``torch.save`` / pickle across that boundary.

Encoding is numpy + json + struct only:

    [4 bytes big-endian header length][utf-8 json header][concatenated array bytes]

The json header carries a free-form ``meta`` dict plus an ordered ``arrays``
list describing each raw block (key, dtype, shape, nbytes). Arrays are written
C-contiguous in the order they appear in ``arrays``.

Large arrays (default >= 256 KiB, typically ``image_frames`` ~27 MiB) can travel
via ``multiprocessing.shared_memory`` instead of the HTTP body. The node and
sidecar must share a POSIX shm namespace (e.g. Docker ``--ipc host``). Semantics
stay bit-identical; this only removes redundant copies of camera tensors. Set
``FD_WIRE_SHM=0`` to force inline payloads.
"""

from __future__ import annotations

import json
import os
import struct
import uuid
from multiprocessing import shared_memory
from typing import Any, Dict, List, Tuple

import numpy as np

_LEN = struct.Struct(">I")
_SHM_THRESHOLD = int(os.environ.get("FD_WIRE_SHM_THRESHOLD", str(256 * 1024)))
_USE_SHM = os.environ.get("FD_WIRE_SHM", "1") not in ("0", "false", "False")


def encode(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> bytes:
    """Serialize a ``(meta, arrays)`` pair to a single bytes buffer.

    Returns the HTTP body. For shm-backed arrays the client must call
    :func:`release_shm` after the round-trip (names are listed in
    ``meta["_shm_owned"]`` when present — also returned via the header meta).
    """
    blocks: List[memoryview] = []
    descriptors = []
    shm_owned: List[str] = []
    for key, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        nbytes = int(a.nbytes)
        use_shm = _USE_SHM and nbytes >= _SHM_THRESHOLD
        if use_shm:
            name = f"fdw_{uuid.uuid4().hex}"
            shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
            try:
                buf = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
                buf[:] = a
            except Exception:
                shm.close()
                shm.unlink()
                raise
            # Unregister from the client's resource_tracker so process exit does
            # not try to unlink a segment the server (or we) already removed.
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:  # noqa: BLE001
                pass
            # Mapping remains until unlink; server attaches by name.
            shm.close()
            shm_owned.append(name)
            descriptors.append(
                {
                    "key": key,
                    "dtype": str(a.dtype),
                    "shape": list(a.shape),
                    "nbytes": nbytes,
                    "shm": name,
                }
            )
        else:
            # memoryview avoids an extra temporary from tobytes() when joining.
            blocks.append(memoryview(a).cast("B"))
            descriptors.append(
                {
                    "key": key,
                    "dtype": str(a.dtype),
                    "shape": list(a.shape),
                    "nbytes": nbytes,
                }
            )
    if shm_owned:
        meta["_shm_owned"] = shm_owned
    header = json.dumps({"meta": meta, "arrays": descriptors}).encode("utf-8")
    inline = b"".join(blocks) if blocks else b""
    return _LEN.pack(len(header)) + header + inline


def decode(buffer: bytes) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Inverse of :func:`encode`.

    Shm-backed arrays are attached and returned as writable ndarray views on the
    shared buffer. Caller should :func:`release_shm` after consuming them (server
    unlinks after each /predict; client unlinks names it created).
    """
    (header_len,) = _LEN.unpack_from(buffer, 0)
    offset = _LEN.size
    header = json.loads(buffer[offset : offset + header_len].decode("utf-8"))
    offset += header_len

    arrays: Dict[str, np.ndarray] = {}
    shm_attached: List[shared_memory.SharedMemory] = []
    for desc in header["arrays"]:
        nbytes = desc["nbytes"]
        shape = tuple(desc["shape"])
        dtype = np.dtype(desc["dtype"])
        if "shm" in desc:
            shm = shared_memory.SharedMemory(name=desc["shm"])
            shm_attached.append(shm)
            arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            # Writable view into shm — torch.from_numpy works; do not mutate.
            arrays[desc["key"]] = arr
        else:
            raw = buffer[offset : offset + nbytes]
            offset += nbytes
            # Single copy into an owned writable buffer.
            arrays[desc["key"]] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    meta = header["meta"]
    # Stash attached handles so release_shm can close them (server side).
    if shm_attached:
        meta = dict(meta)
        meta["_shm_handles"] = shm_attached
    return meta, arrays


def release_shm(meta: Dict[str, Any], *, unlink: bool = False) -> None:
    """Close (and optionally unlink) shm segments referenced by ``meta``.

    - Client after a request: ``release_shm(req_meta, unlink=True)`` for names in
      ``_shm_owned``.
    - Server after predict: close handles in ``_shm_handles`` and unlink those
      names (client may also unlink; double-unlink is ignored).
    """
    handles = meta.pop("_shm_handles", None) or []
    for shm in handles:
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass
    names = list(meta.get("_shm_owned") or [])
    # Also unlink any shm names that appeared only on descriptors (server path).
    if unlink:
        for name in names:
            try:
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                pass
