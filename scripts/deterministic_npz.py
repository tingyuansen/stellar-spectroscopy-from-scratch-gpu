"""Write NumPy archives with stable member order and timestamps."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ whose bytes are reproducible across runs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, values in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(values), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.external_attr = 0o644 << 16
            archive.writestr(member, buffer.getvalue())
