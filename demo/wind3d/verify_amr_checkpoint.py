"""Check split-run equivalence and reject incompatible AMR continuations."""
import csv
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary, runtime = sys.argv[1:]
    count = ctypes.c_int()
    if ctypes.CDLL(runtime).cudaGetDeviceCount(ctypes.byref(count)) or count.value < 1:
        return 77
    with tempfile.TemporaryDirectory(prefix="amr-checkpoint-") as directory:
        root = Path(directory)
        for mode in ("--adaptive", "--amr-uniform"):
            common = [binary, mode, "--grid", "64", "--warmup", "1", "--speed", "12",
                      "--angle", "5", "--cubic", "--no-export", "--amr-compatible-pressure",
                      "--adapt-every", "2"]
            state = root / "state.bin"

            def run(extra, success=True):
                result = subprocess.run(common + extra, capture_output=True, text=True, timeout=180)
                if (result.returncode == 0) != success:
                    raise AssertionError(f"{extra}: {result.stdout}\n{result.stderr}")
                return result

            def rows(path):
                with path.open() as stream:
                    return [{k: v for k, v in row.items() if k != "seconds"} for row in csv.DictReader(stream)]

            full, first, last = [root / f"{mode[2:]}-{part}.csv" for part in ("full", "first", "last")]
            run(["--steps", "8", "--csv", str(full)])
            run(["--steps", "4", "--csv", str(first), "--amr-checkpoint", str(state),
                 "--amr-checkpoint-every", "2"])
            run(["--steps", "8", "--csv", str(last), "--amr-resume", str(state)])
            if rows(full) != rows(first) + rows(last):
                raise AssertionError(f"Split run differs: {mode}")
            for extra in (["--speed", "13"], ["--adapt-every", "3"], ["--max-dt", ".0002"]):
                fresh = root / "rejected.csv"
                run(["--steps", "8", "--csv", str(fresh), "--amr-resume", str(state), *extra], False)
                if fresh.exists():
                    raise AssertionError("Rejected resume created CSV")
            run(["--steps", "8", "--csv", str(first), "--amr-resume", str(state)], False)
            run(["--steps", "4", "--amr-resume", str(state)], False)
            run(["--steps", "8", "--gpus", "2", "--amr-resume", str(state)], False)
            run(["--steps", "8", "--amr-checkpoint", str(state), "--amr-checkpoint-every", "0"], False)
            data = state.read_bytes()
            state.write_bytes(data[:-1])
            run(["--steps", "8", "--amr-resume", str(state)], False)
            print(f"{mode}: exact split continuation and rejection checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
