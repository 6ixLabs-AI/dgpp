"""Compare the production QSA prefill dispatch and its opt-out on a tiny model."""

import array
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


LENGTHS = (127, 128, 129, 256, 513, 1024, 2047, 2048, 2049, 2051, 2052, 2053)
VOCAB = 512


def read_logits(path):
    cases = {}
    with path.open("rb") as stream:
        for count in LENGTHS:
            header = stream.read(8)
            if len(header) != 8 or struct.unpack("=ii", header) != (count, VOCAB):
                raise AssertionError(f"{path}: missing or unexpected case {count}")
            values = array.array("f")
            values.fromfile(stream, count * VOCAB)
            if not all(math.isfinite(v) for v in values):
                raise AssertionError(f"{path}: non-finite logits at {count} rows")
            cases[count] = values
        if stream.read(1):
            raise AssertionError(f"{path}: unexpected trailing output")
    return cases


def compare(warp, partial, require_difference=True):
    changed_cases = 0
    for count in LENGTHS:
        got, want = warp[count], partial[count]
        if count < 128 and got != want:
            raise AssertionError("QSA opt-out changed a prefill below the 128-row threshold")
        changed_cases += count >= 128 and got != want
        worst_l2, worst_gap, flips = 0.0, 0.0, 0
        for row in range(count):
            lo = row * VOCAB
            g, w = got[lo:lo + VOCAB], want[lo:lo + VOCAB]
            numerator = sum((a - b) ** 2 for a, b in zip(g, w))
            denominator = sum(v * v for v in w)
            if denominator == 0:
                raise AssertionError(f"{count} rows, row {row}: degenerate reference")
            rel = math.sqrt(numerator / denominator)
            worst_l2 = max(worst_l2, rel)
            if rel >= 0.01:
                raise AssertionError(f"{count} rows, row {row}: relative L2 {rel} >= 0.01")
            gi = max(range(VOCAB), key=g.__getitem__)
            wi = max(range(VOCAB), key=w.__getitem__)
            if gi != wi:
                flips += 1
                # A changed winner must be within two BF16 ulps of the
                # reference winner, as for the existing Qwen decode checks.
                ulp = math.ldexp(1.0, max(-126, math.frexp(abs(w[wi]))[1] - 1) - 7)
                gap = (w[wi] - w[gi]) / ulp
                worst_gap = max(worst_gap, gap)
                if gap > 2:
                    raise AssertionError(f"{count} rows, row {row}: winner differs by {gap} BF16 ulps")
        print(f"[ OK ] {count} rows: worst relative L2 {worst_l2:.4g}, "
              f"{flips} near-tie winner changes, worst gap {worst_gap:.4g} BF16 ulps")
    if require_difference and not changed_cases:
        raise AssertionError("fixture did not exercise the warp/partial numerical difference")


def main():
    executable = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="dgpp-qsa-prefill-") as temp:
        root = Path(temp)
        cases = {}
        # QwenQsaLayer caches the switch on first use: each arm needs a
        # fresh process, with the rest of the environment held fixed.
        for mode in ("default", "0"):
            output = root / f"warp-{mode}.bin"
            env = dict(os.environ)
            if mode == "default":
                env.pop("DGPP_QSA_WARP", None)
            else:
                env["DGPP_QSA_WARP"] = mode
            subprocess.run([executable, "--qsa-prefill", str(root / "fixture"),
                            "--logits", str(output)],
                           env=env, check=True, timeout=150)
            cases[mode] = read_logits(output)
        # The FP8 fixture pins dispatch sensitivity. NVFP4 expert rounding
        # can hide the attention's small arithmetic differences, so exact
        # agreement is valid there; every per-row numerical gate still holds.
        compare(cases["default"], cases["0"],
                require_difference=os.environ.get("DGPP_TEST_QWEN_NVFP4") != "1")


if __name__ == "__main__":
    main()
