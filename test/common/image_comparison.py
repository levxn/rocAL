# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Phase 0 of the rocAL test-suite revamp: a single, honest pixel comparator.

This is the canonical replacement for the two divergent copies that used to
live at tests/cpp_api/unit_tests/pixel_comparison/image_comparison.py and
tests/python_api/image_comparison.py (those are left untouched; this file
does not modify or depend on either of them). It merges the superset of
their behavior (the python_api copy's one-hot-label comparison) and fixes
two correctness defects in both:

1. The name-based auto-pass is gone. Every case used to be counted PASSED
   without comparison if its name started with Snow/Rain/Jitter/SNPNoise/
   Fog/GaussianNoise/ShotNoise/Spatter, or merely contained the substring
   "Random" -- rocAL's random ops are seeded (rocalSetSeed(0) is called
   unconditionally in unit_tests.cpp), so their output is reproducible and
   comparable like any other op. Cases that genuinely cannot be verified
   today (no golden exists, or an open upstream RPP defect blocks them) are
   now explicit, ticket-tracked entries in skip_list.py instead -- see that
   file for exactly which cases and why.
2. A run that finds zero cases to compare (an empty or missing output
   directory, e.g. from a build that silently failed earlier in the driver
   script) now exits non-zero instead of reporting "0 / 0 passed" and
   exiting 0. This mirrors the same lesson rocm-libraries' RPP test suite
   applies in its own main.cpp: a run that verified nothing is a failure,
   not a vacuous pass.
"""

from PIL import Image
import os
import sys
import datetime
import logging
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skip_list import is_skipped  # noqa: E402


def compare_pixels(img1, img2, aug_name, width, height, image_offset=0):
    pixel_difference = [0, 0, 0, 0, 0, 0]
    # Don't infer color vs grayscale based on filename. Some augmentations (e.g. ColorToGreyscale)
    # are executed on RGB inputs but produce 1-channel outputs.
    sample1 = img1.getpixel((0, 0))
    sample2 = img2.getpixel((0, 0))
    is_color = isinstance(sample1, tuple) and isinstance(sample2, tuple)
    if is_color:
        pixels1 = img1.convert("RGB").load()
        pixels2 = img2.convert("RGB").load()
        channel = 3
    else:
        pixels1 = img1.convert("L").load()
        pixels2 = img2.convert("L").load()
        channel = 1
    total_valid_pixel_count = width * height * channel
    for wt in range(width):
        for ht in range(height):
            ht = ht + image_offset
            if pixels1[wt, ht] != pixels2[wt, ht]:
                if channel == 1:
                    diff_val = abs(pixels1[wt, ht] - pixels2[wt, ht])
                    diff_val = min(diff_val, 5)
                    pixel_difference[diff_val] += 1
                else:
                    for ch in range(channel):
                        diff_val = abs(
                            pixels1[wt, ht][ch] - pixels2[wt, ht][ch])
                        diff_val = min(diff_val, 5)
                        pixel_difference[diff_val] += 1
            else:
                pixel_difference[0] += channel
    return pixel_difference, total_valid_pixel_count


def compare_file_with_list(filename, reference_list):
    with open(filename, 'r') as file:
        file_contents = file.read().strip()
    file_contents = ast.literal_eval(file_contents)
    if len(file_contents) != len(reference_list):
        print("File content and existing list have different lengths.")
        return -1  # Return a negative value to indicate a mismatch in lengths
    mismatch_count = 0
    for file_inner_list, list_inner_list in zip(file_contents, reference_list):
        if file_inner_list != list_inner_list:
            mismatch_count += 1
    return mismatch_count


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    handlers = [
        logging.FileHandler("./rocal_unittest_log_file_" + timestamp + ".log"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    if len(sys.argv) < 3:
        print("Please pass ref_output_folder_path rocal_ouput_folder_path")
        logging.error(
            "Please pass ref_output_folder_path rocal_ouput_folder_path")
        sys.exit(1)
    # Open the two images
    ref_output_path = sys.argv[1]
    rocal_output_path = sys.argv[2]

    if not (os.path.exists(ref_output_path) and os.path.exists(rocal_output_path)):
        logging.error("Path does not Exists")
        sys.exit(1)

    total_case_count = 0
    passed_case_count = 0
    failed_case_count = 0
    skipped_case_count = 0
    failed_case_list = []
    skipped_case_list = []
    golden_output_dir_list = os.listdir(ref_output_path)
    rocal_output_dir_list = os.listdir(rocal_output_path)
    spl_case_meta_data = ["OneHot"]
    golden_file_path = ""
    for aug_name in rocal_output_dir_list:
        temp = aug_name.split(".")
        stem = temp[0]
        file_name_split = stem.split("_")
        if len(file_name_split) > 3:
            golden_file_path = "_".join(file_name_split[:-1]) + ".png"
        else:
            golden_file_path = aug_name

        # Explicit, ticket-tracked skip list (see skip_list.py). Matched against the case's
        # full output stem (e.g. "WarpPerspective_rgb_host"), before any suffix-collapsing, so
        # backend/color-specific entries (e.g. WarpPerspective on host only) resolve correctly.
        # Excluded from both the pass and the fail tally -- a skip is not a silent pass.
        skip_entry = is_skipped(stem)
        if skip_entry is not None:
            skipped_case_count += 1
            skipped_case_list.append(stem)
            logging.info("SKIPPED %s: %s [%s]", stem, skip_entry.reason, skip_entry.ref)
            continue

        if file_name_split[0] in spl_case_meta_data:
            total_case_count = total_case_count + 1
            # Both samples of the last batch come from the labels_folder subdirectory the folder
            # label reader numbers 1, so column 1 is the one that carries the 1.
            reference_list = [[0, 1], [0, 1]]
            rocal_file_path = rocal_output_path + aug_name
            if os.path.exists(rocal_file_path):
                logging.info("Running %s ", aug_name.split(".")[0])
            total_mismatches = compare_file_with_list(rocal_file_path, reference_list)
            if total_mismatches == 0:
                passed_case_count = passed_case_count + 1
                logging.info("PASSED")
            else:
                failed_case_list.append(file_name_split[0])
                failed_case_count = failed_case_count + 1
                logging.info("FAILED")
                logging.info("Printing total mismatches in one hot encode %s", total_mismatches)
        elif golden_file_path in golden_output_dir_list:
            total_case_count = total_case_count + 1
            ref_file_path = ref_output_path + golden_file_path
            rocal_file_path = rocal_output_path + aug_name
            if os.path.exists(rocal_file_path) and os.path.exists(ref_file_path):
                logging.info("Running %s ", aug_name.split(".")[0])
                img1 = Image.open(ref_file_path)
                img2 = Image.open(rocal_file_path)

                # Check if the images have the same dimensions
                if img1.size != img2.size:
                    logging.info(
                        "Golden output and augmentation outputs are having different sizes. Exiting!")
                    sys.exit(1)

                # Compare the pixel values for each image
                pixel_diff = None
                total_count = 0
                if "larger" in aug_name:
                    resize_width = 400
                    resize_height = 400
                    image_offset = 400
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height)
                    pixel_diff2, total_count2 = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixel_diff = [x + y for x,
                                  y in zip(pixel_diff, pixel_diff2)]
                    total_count = total_count + total_count2
                elif "smaller" in aug_name:
                    resize_width = 400
                    resize_height = 400
                    image_offset = 2400
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height)
                    pixel_diff2, total_count2 = compare_pixels(
                        img1, img2, aug_name, resize_width, resize_height, image_offset)
                    pixel_diff = [x + y for x,
                                  y in zip(pixel_diff, pixel_diff2)]
                    total_count = total_count + total_count2
                else:
                    pixel_diff, total_count = compare_pixels(
                        img1, img2, aug_name, img1.size[0], img1.size[1])
                total_pixel_diff = 0
                for pix_diff in range(1, 6):
                    total_pixel_diff += pixel_diff[pix_diff]
                mismatch_percentage = round(
                    (total_pixel_diff / total_count) * 100, 2)
                if ((total_pixel_diff == 0) or (mismatch_percentage < 5.0 and pixel_diff[1] == total_pixel_diff) or       # Ignore test cases with single pixel differences less than 5% of total pixel count
                        (mismatch_percentage < 0.5 and ("Blend" in aug_name or "Rotate" in aug_name) and "hip" in aug_name) or  # Ignore mismatch in rotate augmentation less than 0.5% of total pixel count
                        (mismatch_percentage < 0.5 and ("JpegCompressionDistortion" in aug_name))):  # Ignore mismatch in JpegCompressionDistortion augmentation less than 0.5% of total pixel count
                    passed_case_count = passed_case_count + 1
                    logging.info("PASSED")
                else:
                    failed_case_list.append(golden_file_path)
                    failed_case_count = failed_case_count + 1
                    logging.info("FAILED")
                    logging.info("Printing pixel mismatch %s", pixel_diff)
                    logging.info("Mismatach percentage %0.2f",
                                 mismatch_percentage)
                    for pix_diff in range(1, 6):
                        logging.info("Percentage of %d pixel mismatch %0.2f", pix_diff, round(
                            (pixel_diff[pix_diff] / total_pixel_diff) * 100, 2))
            else:
                failed_case_list.append(golden_file_path)
                failed_case_count = failed_case_count + 1
                logging.info(
                    "Skipping the testcase as file not found %s", rocal_file_path)
        else:
            total_case_count = total_case_count + 1
            failed_case_list.append(golden_file_path)
            failed_case_count = failed_case_count + 1
            logging.info("File not found in ref_output_folder %s",
                         golden_file_path)

    if len(failed_case_list) != 0:
        logging.info("Failing cases: {}".format(", ".join(failed_case_list)))
    if len(skipped_case_list) != 0:
        logging.info("Skipped cases (see skip_list.py for reasons): {}".format(", ".join(skipped_case_list)))
    logging.info(
        "Total case passed --> {} / {} ".format(passed_case_count, total_case_count))
    logging.info(
        "Total case failed --> {} / {} ".format(failed_case_count, total_case_count))
    logging.info(
        "Total case skipped --> {} (not counted above; see skip_list.py)".format(skipped_case_count))

    # A run that verified nothing is a failure, not a vacuous pass -- whether the output
    # directory was empty/missing content, or every case found was skip-listed.
    if total_case_count == 0:
        logging.error(
            "No cases were verified (0 compared, %d skipped). Treating this as a failure: "
            "an empty or entirely-skipped run must not report success.", skipped_case_count)
        sys.exit(1)

    sys.exit(1 if failed_case_count else 0)


if __name__ == "__main__":
    main()
