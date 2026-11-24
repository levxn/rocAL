# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Explicit, ticket-tracked skip list for test/common/image_comparison.py.

Phase 0 of the rocAL test-suite revamp replaces image_comparison.py's old
name-based auto-pass (anything named Snow/Rain/Jitter/SNPNoise/Fog/
GaussianNoise/ShotNoise/Spatter, or containing the substring "Random",
was counted as PASSED without ever opening an image) with a real pixel
comparison for every case, plus this skip list for the specific cases that
are confirmed -- not assumed -- to have no way to pass today.

Every entry here is backed by one of:
  (a) an exhaustive check against the real GoldenOutputsTensor data (the
      golden simply does not exist for that case, under any backend/color
      combination the test driver produces), or
  (b) a currently-open upstream RPP kernel defect, cross-checked against
      RPP's own utilities/test/src/framework/skip_list.hpp and (where cited)
      the referenced fix PR's live merge status.

A skip entry is not a silent pass: image_comparison.py logs
"SKIPPED: <reason> (<ref>)" for every match and excludes it from both the
pass and the fail tally, so `ROCAL_TEST_NO_SKIP_LIST=1` and a quick glance at
the log always show exactly what was not verified and why.

Mirrors the convention in rocm-libraries' RPP test suite
(projects/rpp/utilities/test/src/framework/skip_list.hpp): glob patterns,
one entry per case class, a short reason, a tracking reference, and an
environment-variable override to run the skipped cases anyway and find
entries a fix made obsolete.

Set ROCAL_TEST_NO_SKIP_LIST=1 to disable all matching and run every case for
real -- useful for checking whether an entry here is still deserved.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SkipEntry:
    pattern: str   # fnmatch glob, matched against the case's base name (e.g. "Rain_rgb_host")
    reason: str    # why this case cannot pass today
    ref: str       # tracking ticket/issue, or "needs tracking issue" if none exists yet


# --- Group 1: no golden exists under any backend/color combination -------
# Verified by exhaustive search of GoldenOutputsTensor (239 files, as of this
# writing) for every name/suffix variant these ops are invoked under in
# tests/cpp_api/unit_tests/testAllScripts.sh and tests/python_api/unit_tests.sh.
# These were previously silently auto-passed by image_comparison.py's
# `randomized_augmentation` name list; the list is gone, these are now
# explicit.
_NO_GOLDEN_FIXED_OPS = [
    SkipEntry("Rain_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("SNPNoise_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("GaussianNoise_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("ShotNoise_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("Spatter_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("Jitter_*", "no golden exists for any backend/color combination this op is invoked under",
              "needs tracking issue"),
    SkipEntry("Snow_*",
              "golden exists for Snow_gray_host.png and Snow_rgb_host.png only; the HIP backend and the "
              "FileReader_partial/coco_partial/tfraw-suffixed invocation variants have no golden under any backend",
              "needs tracking issue"),
]

# --- Group 2: the "*Random" code-coverage cases -----------------------------
# These were previously auto-passed by the old comparator's blanket
# `"Random" in file_name_split[0]` substring check, which would have silently
# swallowed any future op whose name happens to contain "Random" even if a
# real golden existed for it. Enumerated explicitly instead of carried
# forward as a glob, specifically so that does not happen again. Every name
# below was verified to have zero matching golden under any backend/color
# combination (see Appendix in test/README.md for the verification command).
_RANDOM_SUFFIXED_CASES = [
    "BlendRandom", "BlurRandom", "BrightnessRandom", "ColorCastRandom", "ColorJitterRandom",
    "ColorTempRandom", "ColorTwistRandom", "ContrastRandom", "CropRandom", "CropResizeRandom",
    "ExposureRandom", "FlipRandom", "FogRandom", "GammaRandom", "GaussianFilterRandom",
    "GaussianNoiseRandom", "HueRandom", "JitterRandom", "JpegCompressionDistortionRandom",
    "NonLinearBlendRandom", "PosterizeRandom", "RandomCrop", "RandomResizedCrop",
    "ResizeCropMirrorRandom", "ResizeMirrorNormalizeRandom", "RotateRandom", "SaturationRandom",
    "ShotNoiseRandom", "SolarizeRandom", "SpatterRandom", "SSDRandomCrop", "VignetteRandom",
    "WarpAffineRandom", "WaterRandom",
]
_NO_GOLDEN_RANDOM_OPS = [
    SkipEntry(f"{name}_*", "no golden exists for any backend/color combination; formerly matched by the "
                           "removed blanket 'Random' substring bypass", "needs tracking issue")
    for name in _RANDOM_SUFFIXED_CASES
]

# --- Group 3: open upstream RPP defects ------------------------------------
# Cross-checked statically against RPP's live skip_list.hpp (rocm-libraries), then
# EMPIRICALLY RE-RUN for real on local gfx1201 hardware against the real
# GoldenOutputsTensor mirror (ROCAL_DATA_PATH=/home/levin/datasets/MIVisionX-data) using
# the unit_tests binary rebuilt from the current rocAL HEAD. Exact commands, raw
# per-pixel-bucket output, and the one case that contradicts its own legacy source
# comment (WarpPerspective, below) are recorded in test/README.md "Validation".
#
# Two of these five op groups came out different from what the legacy, already-disabled
# shell-script comments assumed:
#   - Remap: PASSED on BOTH backends in this run. Removed from the skip list entirely
#     and left enabled for real in both runners -- there is no evidence of a current
#     defect for this op.
#   - WarpPerspective: the legacy comment claims HOST is the broken backend ("a known
#     RPP bug blackens in-bounds output... on the host backend"). The real re-run shows
#     the OPPOSITE: HOST passes cleanly, HIP fails reproducibly (bit-identical output
#     across two independent runs) with every mismatched pixel at the comparator's
#     maximum severity bucket. This is flagged here rather than silently trusted either
#     way; a human should reconcile why the legacy comment and this measurement disagree
#     (likely candidates: RPP/rocAL version drift since the comment was written, or a
#     test-matrix difference) before this entry is treated as final.
_OPEN_RPP_DEFECTS = [
    SkipEntry(
        "ResizeCropMirror_*",
        "RPP's bilinear sampler for resize_crop_mirror is a known defect: RPP's own "
        "skip_list.hpp blanket-skips 'Image_Geometric/ResizeCropMirrorTest.Correctness/*', and the "
        "candidate fix (rocm-libraries PR #10516, 'Fix off-by-one clamp in generic bilinear "
        "interpolation') is still open/draft, unmerged. Nearest-neighbour is not a workaround: the "
        "op rejects it with RPP_ERROR_NOT_IMPLEMENTED. Empirically re-confirmed on gfx1201: HOST "
        "fails for real (0.82% mismatch, dominated by off-by-one but with a 13% max-severity "
        "tail). HIP passes in this run too, but per the prior investigation cited by the legacy "
        "testAllScripts.sh comment, that HIP golden was itself \"minted from the same defective "
        "HIP kernel\" and stops matching once the candidate RPP fix is applied -- i.e. its pass "
        "is not meaningful validation. That specific sub-claim was not independently re-verified "
        "here (it requires building the draft, unmerged RPP fix), so both backends stay skipped "
        "together as the prior investigation concluded.",
        "ROCM-33746, AICV-206",
    ),
    SkipEntry(
        "Phase_*_host",
        "The Phase kernel itself is correct -- it has zero entries in RPP's own skip_list.hpp. This "
        "test's second operand is built via rocalRotateFixed(angle=45, default "
        "ROCAL_LINEAR_INTERPOLATION), which IS an open RPP defect "
        "('Image_Geometric/RotateTest.Correctness/*_a45_BILINEAR' in RPP's skip_list.hpp). "
        "Empirically re-confirmed on gfx1201: HOST fails for real (0.05% mismatch, 93% of it "
        "off-by-one but with a non-zero higher-severity tail); HIP passes cleanly in this run and "
        "is left enabled for real, not skipped. Recommended real fix (out of scope for this "
        "non-invasive phase, requires editing tests/cpp_api/unit_tests/unit_tests.cpp): request "
        "ROCAL_NEAREST_NEIGHBOR_INTERPOLATION for the second operand instead, since RPP's skip "
        "list does not flag nearest-neighbour rotation at this angle.",
        "ROCM-33746",
    ),
    SkipEntry(
        "BitwiseAnd_*_host",
        "Same root cause as Phase_*_host: the BitwiseAnd kernel is correct (absent from RPP's "
        "skip_list.hpp); the failure is inherited from this test's rotate(angle=45, BILINEAR) "
        "second-operand construction, an open RPP defect. Empirically re-confirmed on gfx1201: "
        "HOST fails for real (0.06% mismatch, 67% off-by-one, 25% at higher severity buckets); "
        "HIP passes cleanly in this run and is left enabled for real, not skipped.",
        "ROCM-33746",
    ),
    SkipEntry(
        "BitwiseOr_*_host",
        "Same root cause as Phase_*_host/BitwiseAnd_*_host. Empirically re-confirmed on gfx1201: "
        "HOST fails for real (0.05% mismatch); HIP passes cleanly and is left enabled.",
        "ROCM-33746",
    ),
    SkipEntry(
        "BitwiseXor_*_host",
        "Same root cause as Phase_*_host/BitwiseAnd_*_host. Empirically re-confirmed on gfx1201: "
        "HOST fails for real (0.08% mismatch); HIP passes cleanly and is left enabled.",
        "ROCM-33746",
    ),
    SkipEntry(
        "WarpPerspective_*_hip",
        "The legacy unit_tests.sh/testAllScripts.sh comment claims HOST is the broken backend "
        "(\"a known RPP bug blackens in-bounds output... on the host backend\") and only ever ran "
        "this op on HIP. The real, empirical re-run on gfx1201 shows the OPPOSITE: WarpPerspective "
        "passes cleanly on HOST and fails reproducibly on HIP (bit-identical output across two "
        "independent runs; 0.21% mismatch, 100% of mismatched pixels at the comparator's maximum "
        "severity bucket -- a large, structural difference, not rounding noise). This is recorded "
        "as a real, current, empirical measurement; the discrepancy with the legacy comment is "
        "flagged rather than silently resolved either way and should be reconciled by a human "
        "(likely candidates: RPP/rocAL version drift since the legacy comment was written, since "
        "RPP's own skip_list.hpp still lists WarpPerspectiveTest.Correctness HOST identity/shift "
        "BILINEAR entries that no longer reproduce here; or a difference in exactly which "
        "perspective matrix/ROI/size this test exercises versus RPP's own grid). HOST is left "
        "enabled for real, not skipped, since it measurably passes today.",
        "needs tracking issue",
    ),
]

SKIP_LIST = _NO_GOLDEN_FIXED_OPS + _NO_GOLDEN_RANDOM_OPS + _OPEN_RPP_DEFECTS

_DISABLED_ENV_VAR = "ROCAL_TEST_NO_SKIP_LIST"


def skip_list_disabled() -> bool:
    """True when ROCAL_TEST_NO_SKIP_LIST=1 is set, i.e. every case should run for real."""
    return os.environ.get(_DISABLED_ENV_VAR) == "1"


def is_skipped(name: str) -> Optional[SkipEntry]:
    """Returns the matching SkipEntry for `name` (a case's base name, e.g. "Rain_rgb_host"),
    or None if `name` is not skip-listed or ROCAL_TEST_NO_SKIP_LIST=1 is set."""
    if skip_list_disabled():
        return None
    for entry in SKIP_LIST:
        if fnmatch.fnmatch(name, entry.pattern):
            return entry
    return None
