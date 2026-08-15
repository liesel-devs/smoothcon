# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem

"""Regenerate committed mgcv oracle assets without adding R to pytest.

Run this file directly. It installs the requested local mgcv source into a temporary
R library, invokes the adjacent R generator, and writes compressed NumPy assets plus a
checksum manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _metadata(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        result[key] = int(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgcv-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "assets")
    args = parser.parse_args()
    source = args.mgcv_source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="smoothcon-mgcv-") as temporary:
        root = Path(temporary)
        library = root / "library"
        source_copy = root / "mgcv"
        csv_output = root / "csv"
        library.mkdir()
        shutil.copytree(source, source_copy)
        # Some macOS R installations define ENABLE_NLS without shipping the gettext
        # development headers. NLS only affects translated error messages and is not
        # involved in any oracle calculation, so disable it in the temporary copy.
        general_header = source_copy / "src" / "general.h"
        general_header.write_text(
            general_header.read_text().replace("#ifdef ENABLE_NLS", "#if 0", 1)
        )
        install_expression = (
            f"install.packages({str(source_copy)!r}, repos=NULL, type='source', "
            f"lib={str(library)!r}); "
            f"if (!requireNamespace('mgcv', lib.loc={str(library)!r}, "
            "quietly=TRUE)) quit(status=1)"
        )
        subprocess.run(["Rscript", "-e", install_expression], check=True)
        environment = os.environ.copy()
        environment["R_LIBS"] = str(library)
        subprocess.run(
            ["Rscript", str(HERE / "generate.R"), str(csv_output)],
            check=True,
            env=environment,
        )

        cases: dict[str, dict[str, object]] = {}
        for directory in sorted(path for path in csv_output.iterdir() if path.is_dir()):
            arrays = {
                "x": _matrix(directory / "x.csv"),
                "new_x": _matrix(directory / "new_x.csv"),
                "basis": _matrix(directory / "basis.csv"),
                "penalty": _matrix(directory / "penalty.csv"),
                "new_basis": _matrix(directory / "new_basis.csv"),
            }
            for name in (
                "transformed_basis",
                "transformed_penalty",
                "transformed_new_basis",
            ):
                path = directory / f"{name}.csv"
                if path.exists():
                    arrays[name] = _matrix(path)
            target = output / f"{directory.name}.npz"
            np.savez_compressed(target, **arrays)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            cases[directory.name] = {
                **_metadata(directory / "metadata.txt"),
                "shapes": {key: list(value.shape) for key, value in arrays.items()},
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }

        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        manifest = {
            "mgcv_version": (csv_output / "version.txt").read_text().strip(),
            "mgcv_commit": commit,
            "generator": "generate.R",
            "cases": cases,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
