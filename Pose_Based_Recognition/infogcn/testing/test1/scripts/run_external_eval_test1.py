"""External evaluation runner for testing/test1 reusing processed New40 features on GPU."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_test1 import NEW40_MANIFEST_PATH, require_cuda  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EXP10 InfoGCN external evaluation on existing New40 processed manifest"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=NEW40_MANIFEST_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing testing/test1/results outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device, gpu_name = require_cuda()
    print(f"Pre-flight CUDA check passed: {device} ({gpu_name})")

    if not args.manifest.exists():
        raise FileNotFoundError(f"Processed New40 manifest not found: {args.manifest}")

    cmd = [
        sys.executable,
        "-B",
        str(SCRIPT_DIR / "evaluate_exp10_test1.py"),
        "--manifest",
        str(args.manifest),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    print(f">>> Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print("\nExternal evaluation completed successfully.")


if __name__ == "__main__":
    main()
