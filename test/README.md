# `test/` — rocAL Phase 0 test suite

This directory is the output of Phase 0 ("make CI able to fail") of the rocAL test-suite
revamp. It does not modify, depend on internals of, or replace `tests/` — see `TESTING.md` at
the repository root for the relationship between the two trees and the full policy this
directory follows.

## Layout

```
test/
  common/
    image_comparison.py   canonical pixel comparator (no name-based auto-pass; see docstring)
    skip_list.py           explicit, ticket-tracked skip list (see docstring)
  cpp_api/
    run_unit_tests.sh       fail-fast sibling of tests/cpp_api/unit_tests/testAllScripts.sh
  python_api/
    run_unit_tests.sh       fail-fast sibling of tests/python_api/unit_tests.sh
```

Both runner scripts build/invoke the **unmodified** sources under `tests/` (the same
`unit_tests.cpp`/`CMakeLists.txt`, the same `unit_test.py`) — nothing in `tests/` is edited.
The only two differences from the legacy drivers are: `set -e -o pipefail` (a failure aborts the
run instead of being silently swallowed), and the final comparison call uses
`test/common/image_comparison.py` instead of the legacy per-tree copies.

## Running

```bash
export ROCAL_DATA_PATH=/path/to/MIVisionX-data   # must contain a rocal_data/ directory

test/cpp_api/run_unit_tests.sh 2 2      # device: 0=host 1=hip 2=both; color: 0=gray 1=rgb 2=both
test/python_api/run_unit_tests.sh 2 2
```

## Why five previously-disabled ops are re-enabled here

`testAllScripts.sh`/`unit_tests.sh` fully comment out `ResizeCropMirror`, `Phase`,
`BitwiseAnd`/`BitwiseOr`/`BitwiseXor`, `Remap`, and `WarpPerspective`-on-host, citing open RPP
defects (`ROCM-33746`/`AICV-206`). This suite re-enables all five so they actually execute; their
verdicts are handled by `test/common/skip_list.py`, not by never running them. Two came out
different from what the legacy comments assumed — `Remap` passes cleanly on both backends today
(no longer skipped), and `WarpPerspective` fails on HIP rather than HOST (the opposite of the
legacy claim). See `TESTING.md`'s "Validation evidence" section at the repository root for the
full real-hardware re-run that produced these numbers, including the exact per-pixel-bucket
mismatch data for every case that still fails.

## Verifying the `*Random` skip-list enumeration

The ~33 explicitly-enumerated `*Random`-suffixed entries in `skip_list.py` (replacing the old
blanket `"Random" in name` substring bypass) were derived from the actual invocation matrix and
verified against the real golden set with:

```bash
cd tests/cpp_api/unit_tests
grep -oE '\$\{output_path\}[A-Za-z0-9]*Random[A-Za-z0-9]*' testAllScripts.sh \
  | sed 's/\${output_path}//' | sort -u

cd "$ROCAL_DATA_PATH/rocal_data/GoldenOutputsTensor"
for name in <names from above>; do ls | grep -i "^${name}_" || echo "$name: no golden"; done
```

`tests/python_api/unit_tests.sh` produces no `*Random`-suffixed output names at all (verified by
the same grep against that file).
