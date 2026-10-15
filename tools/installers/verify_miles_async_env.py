"""Audit real installed artifacts, native source pins and Miles eager imports."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "profile-dir",
        "espnet-source",
        "sglang-source",
        "miles-source",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    profile_dir = Path(args.profile_dir)
    profile = json.loads((profile_dir / "profile.json").read_text())
    sys.path.insert(0, args.miles_source)
    result = {
        "source_pins": {},
        "packages": {},
        "imports": {},
        "errors": [],
        "miles_source_only": True,
        "gpu_execution": False,
    }
    for name in ("espnet", "sglang", "miles"):
        path = Path(getattr(args, name + "_source"))
        head = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        result["source_pins"][name] = head
        if profile["source_pins"].get(name, head) != head:
            result["errors"].append("Source pin differs: " + name)
        if subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip():
            result["errors"].append("Source changed: " + name)
    for line in (profile_dir / "requirements.txt").read_text().splitlines():
        if " @ " not in line:
            continue
        name, value = line.split(" @ ", 1)
        _, expected = value.split(" --hash=sha256:")
        try:
            distribution = importlib.metadata.distribution(name.split("[", 1)[0])
            direct = json.loads(distribution.read_text("direct_url.json"))
            actual = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
            result["packages"][name] = {
                "version": distribution.version,
                "expected_sha256": expected,
                "recorded_sha256": actual,
                "url": direct.get("url"),
            }
            if actual != expected:
                result["errors"].append("Artifact hash differs: " + name)
        except Exception as error:
            result["errors"].append(name + ": " + repr(error))
    for name in (
        "sglang_router",
        "polars",
        "pylatexenc",
        "sglang.srt.entrypoints.engine",
        "miles.utils.data",
        "miles.ray.train_actor",
        "miles.rollout.rm_hub",
        "espnet2.rl.miles_components",
        "espnet2.rl.miles_async_driver",
    ):
        try:
            module = importlib.import_module(name)
            result["imports"][name] = {
                "ok": True,
                "file": getattr(module, "__file__", None),
            }
        except Exception:
            result["imports"][name] = {"ok": False, "error": traceback.format_exc()}
    try:
        path = Path(args.miles_source) / "train_async.py"
        spec = importlib.util.spec_from_file_location(
            "actual_pinned_miles_train_async", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result["imports"]["train_async.py"] = {
            "ok": True,
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    except Exception:
        result["imports"]["train_async.py"] = {
            "ok": False,
            "error": traceback.format_exc(),
        }
    check = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    result.update(
        pip_check_exit_code=check.returncode,
        pip_check_stdout=check.stdout,
        pip_check_stderr=check.stderr,
        artifact_count=len(result["packages"]),
    )
    result["passed"] = (
        not result["errors"]
        and check.returncode == 0
        and all(v["ok"] for v in result["imports"].values())
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "packages"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
