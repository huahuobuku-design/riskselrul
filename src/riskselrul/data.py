"""C-MAPSS provenance, structural checks, and safe extraction from a supplied ZIP."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from importlib.resources import files
import zipfile
import numpy as np

DATASETS = ("FD001", "FD003")


def inspect_data(data_root, dataset="FD001", require_known_hashes=False):
    if dataset not in DATASETS:
        raise ValueError(f"Supported dataset profiles: {DATASETS}")
    manifest = json.loads(files("riskselrul").joinpath("assets/data_manifest.json").read_text(encoding="utf-8"))
    reports = []
    for prefix in ("train", "test", "RUL"):
        name = f"{prefix}_{dataset}.txt"
        path = Path(data_root)/name
        if not path.is_file():
            raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
        array = np.loadtxt(path, ndmin=1 if prefix == "RUL" else 2)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite value in {name}")
        if prefix == "RUL":
            if array.shape != (100,) or np.any(array < 0):
                raise ValueError(f"{name} must contain 100 nonnegative endpoint labels")
        else:
            if array.shape[1] != 26 or not np.array_equal(np.unique(array[:, 0]), np.arange(1, 101)):
                raise ValueError(f"{name} must have 26 columns and engine IDs 1..100")
            if not np.array_equal(array[:, :2], array[:, :2].astype(int)):
                raise ValueError(f"Noninteger engine or cycle in {name}")
            expected_order = np.lexsort((array[:, 1], array[:, 0]))
            if not np.array_equal(expected_order, np.arange(len(array))):
                raise ValueError(f"Engine/cycle rows must be ordered in {name}")
            for engine in range(1, 101):
                cycles = array[array[:, 0] == engine, 1]
                if not np.array_equal(cycles, np.arange(1, len(cycles)+1)):
                    raise ValueError(f"Nonconsecutive cycles for engine {engine} in {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = digest == manifest[name]["sha256"]
        if require_known_hashes and not matches:
            raise ValueError(f"File differs from archived NASA bytes: {name}")
        reports.append(dict(filename=name, shape=list(array.shape), sha256=digest, matches_archived_bytes=matches))
    return dict(dataset=dataset, simulated_data=True, files=reports)


def extract_cmapss(zip_path, destination):
    """Extract only the six known dataset members; never overwrite existing files."""
    destination = Path(destination)
    allowed = {f"{prefix}_{ds}.txt" for ds in DATASETS for prefix in ("train", "test", "RUL")}
    with zipfile.ZipFile(zip_path) as archive:
        selected = {}
        for member in archive.infolist():
            normalized = member.filename.replace("\\", "/")
            path = PurePosixPath(normalized)
            if path.is_absolute() or ".." in path.parts or ":" in normalized or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP member")
            if path.name in allowed:
                if path.name in selected or member.file_size > 32_000_000:
                    raise ValueError("Duplicate or oversized dataset member")
                selected[path.name] = member
        if set(selected) != allowed:
            raise ValueError("ZIP does not contain all six FD001/FD003 files")
        for name in selected:
            if (destination/name).exists():
                raise FileExistsError(f"Refusing to overwrite {name}")
        contents = {name: archive.read(member) for name, member in selected.items()}
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        with (destination/name).open("xb") as stream:
            stream.write(content)
    return [str(destination/name) for name in sorted(contents)]
