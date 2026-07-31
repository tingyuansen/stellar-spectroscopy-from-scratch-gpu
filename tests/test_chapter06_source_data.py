"""Exact source and static-table staging gates for Chapter 6."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest

import numpy as np

from payne_zero_atmosphere.line_profile_math import (
    build_voigt_profile_basis,
    load_line_opacity_tables,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "9c44001feae40b85146630499e6f8a5fed42e5af"

SOURCE_HASHES = {
    "src/payne_zero_atmosphere/line_profile_math.py": (
        "9a5794140f00ff3c3fb6c2e3b28461bbc22b471f962d275055c066ad7f8acd15"
    ),
    "src/payne_zero_atmosphere/line_catalog.py": (
        "2ad08f866ae1917a8327d2cf54115d7afe407a5e09645bb12b70e762082d1d92"
    ),
    "src/payne_zero_atmosphere/hydrogen_line_profile.py": (
        "6a48f43afee9e326d2f86282f22f44f5654e243a335cc0490c99f86c41451be0"
    ),
    "src/payne_zero_atmosphere/line_opacity.py": (
        "d0f9c43919be58a42547e12b7abc22161a7558bf17abbcd375ab04ccf57d7cc6"
    ),
    "src/payne_zero_synthesis/atomic_lines.py": (
        "0fa52833fb16487da1d5bfaaf5628a46751f888c1a57894a5037daa6d6667ab0"
    ),
    "src/payne_zero_synthesis/line_opacity.py": (
        "639b95c3812f1a7d227b797fa89a4d6ef9725d5f0e1284f3d49cf86844278275"
    ),
}

ATMOSPHERE_TABLE = (
    REPOSITORY_ROOT / "data/static/atmosphere_tables/line_opacity_tables.npz"
)
SYNTHESIS_TABLE = (
    REPOSITORY_ROOT / "data/static/synthesis_tables/line_profile_tables.npz"
)
TABLE_HASHES = {
    ATMOSPHERE_TABLE: (
        "89f486122cb8939b23dc5423145a46d88a77df8daf57a1def35055b7b8205f16"
    ),
    SYNTHESIS_TABLE: (
        "87b47fc76bed10455218f43c4b6686525b961002e72d6a5ef01255a08deb27d4"
    ),
}

ATMOSPHERE_TABLE_SHAPES = {
    "voigt_interpolation_table": (81,),
    "hydrogen_profile_table": (81,),
}
SYNTHESIS_TABLE_SHAPES = {
    "hydrogen_continuum_edges": (15,),
    "radiative_damping_sums": (96,),
    "impact_electron_density_thresholds_cm3": (2, 2),
    "stark_knm_table": (4, 3),
    "stark_probability_table": (7, 5, 15),
    "stark_wing_correction_c": (5, 7),
    "stark_wing_correction_d": (5, 7),
    "stark_pressure_grid": (5,),
    "stark_beta_grid": (15,),
    "harris_profile_h0_table": (2001,),
    "harris_profile_h1_table": (2001,),
    "harris_profile_h2_table": (2001,),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 identity of one local artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class Chapter06SourceDataTests(unittest.TestCase):
    """Freeze the dependency-complete ordinary-line source boundary."""

    def test_source_files_and_source_manifest_have_exact_identities(self) -> None:
        source_manifest = json.loads(
            (REPOSITORY_ROOT / "src/PAYNE_ZERO_SOURCE_MANIFEST.json").read_text()
        )
        self.assertEqual(source_manifest["payne_zero_commit"], PINNED_COMMIT)
        entries = {entry["local_path"]: entry for entry in source_manifest["entries"]}
        for relative_path, expected_hash in SOURCE_HASHES.items():
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                self.assertEqual(sha256(path), expected_hash)
                entry = entries[relative_path]
                self.assertEqual(entry["copy_mode"], "byte-identical file")
                self.assertEqual(entry["source_file_sha256"], expected_hash)

        hydrogen_entry = entries["src/payne_zero_atmosphere/hydrogen_line_profile.py"]
        self.assertIn("Import dependency only", hydrogen_entry["scope"])
        self.assertIn("deferred to Chapter 7", hydrogen_entry["scope"])

    def test_static_archives_have_exact_bytes_keys_shapes_and_dtypes(self) -> None:
        for path, expected_hash in TABLE_HASHES.items():
            with self.subTest(path=path.name):
                self.assertEqual(sha256(path), expected_hash)

        for path, expected_shapes in (
            (ATMOSPHERE_TABLE, ATMOSPHERE_TABLE_SHAPES),
            (SYNTHESIS_TABLE, SYNTHESIS_TABLE_SHAPES),
        ):
            with (
                self.subTest(path=path.name),
                np.load(path, allow_pickle=False) as archive,
            ):
                self.assertEqual(set(archive.files), set(expected_shapes))
                for name, shape in expected_shapes.items():
                    values = np.asarray(archive[name])
                    self.assertEqual(values.shape, shape)
                    self.assertEqual(values.dtype, np.dtype("float64"))

    def test_data_manifest_owns_both_archives_and_every_member(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / "data/MANIFEST.json").read_text())
        entries = {entry["path"]: entry for entry in manifest["entries"]}
        for path, expected_hash in TABLE_HASHES.items():
            relative_path = str(path.relative_to(REPOSITORY_ROOT))
            with (
                self.subTest(path=relative_path),
                np.load(path, allow_pickle=False) as archive,
            ):
                entry = entries[relative_path]
                self.assertEqual(entry["role"], "static")
                self.assertEqual(entry["source_commit"], PINNED_COMMIT)
                self.assertEqual(entry["source_sha256"], expected_hash)
                self.assertEqual(entry["sha256"], expected_hash)
                self.assertFalse(entry["requires_optional_full_catalog"])
                self.assertEqual(set(entry["arrays"]), set(archive.files))
                for name in archive.files:
                    member = entry["arrays"][name]
                    self.assertEqual(member["shape"], list(archive[name].shape))
                    self.assertEqual(member["dtype"], str(archive[name].dtype))
                    self.assertTrue(member["unit"])
                    self.assertEqual(len(member["axes"]), archive[name].ndim)
                    self.assertTrue(member["ownership"])

    def test_all_six_modules_import_without_deferred_data_or_upstream_tree(
        self,
    ) -> None:
        modules = (
            "payne_zero_atmosphere.line_profile_math",
            "payne_zero_atmosphere.line_catalog",
            "payne_zero_atmosphere.hydrogen_line_profile",
            "payne_zero_atmosphere.line_opacity",
            "payne_zero_synthesis.atomic_lines",
            "payne_zero_synthesis.line_opacity",
        )
        script = "\n".join(
            (
                "from pathlib import Path",
                "import importlib",
                f"root = Path({str(REPOSITORY_ROOT / 'src')!r}).resolve()",
                f"names = {modules!r}",
                "for name in names:",
                "    module = importlib.import_module(name)",
                "    Path(module.__file__).resolve().relative_to(root)",
            )
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        for name in (
            "PAYNE_ZERO_DATA_ROOT",
            "PAYNE_ZERO_SOURCE_CATALOG_ROOT",
            "PAYNE_ZERO_SYNTHESIS_SOURCE_CATALOG_ROOT",
        ):
            environment.pop(name, None)
        subprocess.run(
            [os.sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_atmosphere_and_synthesis_harris_authorities_remain_distinct(
        self,
    ) -> None:
        load_line_opacity_tables(ATMOSPHERE_TABLE, force_reload=True)
        build_voigt_profile_basis.cache_clear()
        atmosphere = build_voigt_profile_basis()

        with np.load(SYNTHESIS_TABLE, allow_pickle=False) as archive:
            synthesis_h0 = np.asarray(archive["harris_profile_h0_table"])
            synthesis_h1 = np.asarray(archive["harris_profile_h1_table"])
            synthesis_h2 = np.asarray(archive["harris_profile_h2_table"])

        self.assertFalse(np.array_equal(atmosphere.gaussian_profile, synthesis_h0))
        self.assertFalse(np.array_equal(atmosphere.first_correction, synthesis_h1))
        self.assertFalse(np.array_equal(atmosphere.second_correction, synthesis_h2))
        self.assertEqual(
            np.flatnonzero(atmosphere.first_correction != synthesis_h1).tolist(),
            list(range(21, 40)),
        )
        h1_difference = np.abs(atmosphere.first_correction - synthesis_h1)
        self.assertEqual(int(np.argmax(h1_difference)), 30)
        self.assertEqual(float(np.max(h1_difference)), 0.005044391583364227)
        self.assertEqual(
            np.flatnonzero(atmosphere.gaussian_profile != synthesis_h0).tolist(),
            [1721],
        )
        self.assertEqual(
            np.flatnonzero(atmosphere.second_correction != synthesis_h2).tolist(),
            [1721],
        )


if __name__ == "__main__":
    unittest.main()
