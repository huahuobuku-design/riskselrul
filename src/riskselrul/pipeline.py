"""Run the migrated WP05A benchmark in a separate CPU process."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.resources import files


def reproduce(data_root, output, dataset="FD001", seed=42, streams=(9001,), bootstrap_reps=2000):
    from .data import inspect_data
    from .audit import audit_run
    if bootstrap_reps < 1 or seed < 0 or not streams:
        raise ValueError("Positive bootstrap repetitions, nonnegative seed, and test streams are required")
    missing = [name for name in ("numpy", "pandas", "scipy", "sklearn", "threadpoolctl") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Install riskselrul[train]; missing: " + ", ".join(missing))
    data_root = Path(data_root).resolve()
    input_report = inspect_data(data_root, dataset)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; historical runs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = files("riskselrul").joinpath("_legacy/config.json").read_bytes()
    config = dict(profile="wp05a-v2-migrated", dataset=dataset, seed=seed, streams=list(streams),
                  bootstrap_reps=bootstrap_reps, data=input_report,
                  legacy_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                  python=sys.version, platform=platform.platform(),
                  started_utc=datetime.now(timezone.utc).isoformat())
    from importlib.metadata import version
    config["versions"] = {name:version(name) for name in ("numpy","pandas","scipy","scikit-learn","threadpoolctl")}
    (output/"package_run_manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [sys.executable, "-m", "riskselrul._legacy.train_entry", "--data-root", str(data_root),
               "--out", str(output), "--ds", dataset, "--seeds", str(seed), "--streams", *map(str, streams),
               "--reps", str(bootstrap_reps)]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    # No shell expansion, automatic downloads, dynamic installs, or object deserialization.
    with (output/"train.stdout.log").open("w", encoding="utf-8") as stdout, (output/"train.stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, env=environment, text=True, encoding="utf-8")
        for line in process.stdout:
            print(line, end="", flush=True)
            stdout.write(line)
            stdout.flush()
        code = process.wait()
    (output/"train.exitcode.txt").write_text(str(code), encoding="ascii")
    if code:
        raise RuntimeError(f"Training exited {code}; see logs in {output}")
    report, direction = audit_run(output)
    (output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (output/"direction_common_denominator.json").write_text(json.dumps(direction, indent=2, allow_nan=False), encoding="utf-8")
    if report["mismatches"]:
        raise RuntimeError("Independent audit failed")
    digests = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(output.iterdir()) if f.is_file()}
    (output/"output_hashes.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="REPRODUCTION_PASS", output=str(output), **report)), flush=True)
    return report
