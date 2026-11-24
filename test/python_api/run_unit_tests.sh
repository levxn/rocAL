#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Phase 0 of the rocAL test-suite revamp: a fail-fast, non-invasive sibling of
# tests/python_api/unit_tests.sh.
#
# This script does NOT modify tests/python_api/ in any way -- it invokes the unmodified
# tests/python_api/unit_test.py with the exact same invocation matrix as unit_tests.sh,
# with three differences:
#
#   1. `set -e -o pipefail` -- any failing invocation now aborts the script loudly instead
#      of being silently swallowed until the final comparator call reports a confusing,
#      indirect failure.
#   2. The final comparison call uses test/common/image_comparison.py instead of
#      tests/python_api/image_comparison.py -- the new comparator has no name-based
#      auto-pass and uses the explicit skip list in test/common/skip_list.py instead
#      (see that file for exactly which cases are skipped and why).
#   3. The five op groups that unit_tests.sh fully comments out (resize_crop_mirror,
#      phase, bitwise_and/or/xor, remap, warp_perspective-on-host) are re-enabled here
#      so they actually execute. Their expected-to-fail verdicts are handled by the skip
#      list, not by never running them -- see test/README.md "Validation" for the real
#      local re-run that confirmed each is still blocked by an open upstream RPP defect.

set -e -o pipefail

cwd="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
unit_test_py="$cwd/../../tests/python_api/unit_test.py"
comparator="$cwd/../common/image_comparison.py"

if [[ $ROCAL_DATA_PATH == "" ]]
then
    echo "Need to export ROCAL_DATA_PATH"
    exit
fi

# Path to inputs and outputs
one_hot_data_path=${ROCAL_DATA_PATH}/rocal_data/images_jpg/labels_folder/
image_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
web_dataset_path=${ROCAL_DATA_PATH}/rocal_data/web_dataset/tar_file/
coco_detection_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/images/
coco_json_path=${ROCAL_DATA_PATH}/rocal_data/coco/coco_10_img/annotations/coco_data.json
tf_classification_path=${ROCAL_DATA_PATH}/rocal_data/tf/classification/
tf_detection_path=${ROCAL_DATA_PATH}/rocal_data/tf/detection/
caffe_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe/classification/
caffe_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe/detection/
caffe2_classification_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/classification/
caffe2_detection_path=${ROCAL_DATA_PATH}/rocal_data/caffe2/detection/
mxnet_path=${ROCAL_DATA_PATH}/rocal_data/mxnet/
output_path=./rocal_python_unittest_output_folder_$(date +%Y-%m-%d_%H-%M-%S)/
golden_output_path=${ROCAL_DATA_PATH}/rocal_data/GoldenOutputsTensor/

display=0
batch_size=2
device=0
width=416
height=416
device_name="host"
rgb_name=("gray" "rgb")
rgb=1
dev_start=0
dev_end=1
rgb_start=0
rgb_end=1

# python version
ver=$(python3 -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";)

if [ "$#" -gt 0 ]; then
    if [ "$1" -eq 0 ]; then # For only HOST backend
        dev_start=0
        dev_end=0
    elif [ "$1" -eq 1 ]; then # For only HIP backend
        dev_start=1
        dev_end=1
    elif [ "$1" -eq 2 ]; then # For both HOST and HIP backend
        dev_start=0
        dev_end=1
    fi
fi

if [ "$#" -gt 1 ]; then
    if [ "$2" -eq 0 ]; then # For only Greyscale inputs
        rgb_start=0
        rgb_end=0
    elif [ "$2" -eq 1 ]; then # For only RGB inputs
        rgb_start=1
        rgb_end=1
    elif [ "$2" -eq 2 ]; then # For both RGB and Greyscale inputs
        rgb_start=0
        rgb_end=1
    fi
fi

mkdir "$output_path"

for ((device=dev_start;device<=dev_end;device++))
do
    if [ $device -eq 1 ]
    then
        device_name="hip"
        backend_arg=rocal-gpu
        echo "Running HIP Backend..."
    else
        backend_arg=no-rocal-gpu
        echo "Running HOST Backend..."
    fi
    for ((rgb=rgb_start;rgb<=rgb_end;rgb++))
    do
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name lens_correction --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}LensCorrection_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name exposure --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Exposure_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name flip --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Flip_${rgb_name[$rgb]}_${device_name}"

        # web dataset reader
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name gamma_correction --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name blur --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Blur_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name brightness --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Brightness_${rgb_name[$rgb]}_${device_name}"

        # coco detection
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name gamma_correction --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Gamma_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name contrast --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Contrast_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name vignette --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Vignette_${rgb_name[$rgb]}_${device_name}"

        # tf classification
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name blend --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Blend_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name warp_affine --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}WarpAffine_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name blur --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Blur_${rgb_name[$rgb]}_${device_name}"

        # tf detection
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name snp_noise --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}SNPNoise_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name color_temp --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ColorTemp_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name fog --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Fog_${rgb_name[$rgb]}_${device_name}"

        # caffe classification
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name rotate --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Rotate_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name brightness --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Brightness_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name hue --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Hue_${rgb_name[$rgb]}_${device_name}"

        # caffe detection
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name saturation --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Saturation_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name color_twist --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ColorTwist_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name rain --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Rain_${rgb_name[$rgb]}_${device_name}"

        # caffe2 classification
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name center_crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}"
        # Re-enabled (was commented out in unit_tests.sh under ROCM-33746/AICV-206). Expected to
        # be caught by the "ResizeCropMirror_*" skip-list entry -- see
        # test/common/skip_list.py and test/README.md "Validation".
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name resize_crop_mirror --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ResizeCropMirror_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name snow --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Snow_${rgb_name[$rgb]}_${device_name}"

        # caffe2 detection
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name fish_eye --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}FishEye_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name pixelate --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Pixelate_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name center_crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropCenter_${rgb_name[$rgb]}_${device_name}_cmn"

        # mxnet
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name jitter --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Jitter_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name resize_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}ResizeMirrorNormalize_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet"

        # CMN
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_FileReader"
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_webDataset"
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_coco"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_tfDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffeDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Classification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_caffe2Detection"
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop_mirror_normalize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}CropMirrorNormalize_${rgb_name[$rgb]}_${device_name}_mxnet"

        # crop
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_FileReader"
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_webDataset"
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_coco"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_tfDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffeDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Classification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_caffe2Detection"
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name crop --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb  --$backend_arg -f "${output_path}Crop_${rgb_name[$rgb]}_${device_name}_mxnet"

        # resize
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_default_FileReader"
        python"$ver" "$unit_test_py" --image-dataset-path "$web_dataset_path" --reader-type web_dataset --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 1 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_stretch_webDataset"
        python"$ver" "$unit_test_py" --image-dataset-path "$coco_detection_path" --reader-type coco --json-path "$coco_json_path" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 1 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_stretch_coco"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_classification_path" --reader-type "tf_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 2 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notsmaller_tfClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$tf_detection_path" --reader-type "tf_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 1 --scaling-mode 3 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bilinear_notlarger_tfDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_classification_path" --reader-type "caffe_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 2 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_bicubic_default_caffeClassification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe_detection_path" --reader-type "caffe_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 0 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_nearestneighbor_default_caffeDetection"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_classification_path" --reader-type "caffe2_classification" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 3 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_lanczos_default_caffe2Classification"
        python"$ver" "$unit_test_py" --image-dataset-path "$caffe2_detection_path" --reader-type "caffe2_detection" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 5 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_triangular_default_caffe2Detection"
        python"$ver" "$unit_test_py" --image-dataset-path "$mxnet_path" --reader-type "mxnet" --augmentation-name resize --batch-size $batch_size  --max-width $width --max-height $height --color-format $rgb --interpolation-type 4 --scaling-mode 0 --$backend_arg -f "${output_path}Resize_${rgb_name[$rgb]}_${device_name}_gaussian_default_mxnet"

        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name color_cast --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}ColorCast_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name grid_mask --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}GridMask_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name median_filter --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}MedianFilter_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name gaussian_filter --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}GaussianFilter_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name non_linear_blend --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}NonLinearBlend_${rgb_name[$rgb]}_${device_name}"

        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name dilate --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Dilate_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name erode --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Erode_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name magnitude --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Magnitude_${rgb_name[$rgb]}_${device_name}"
        # Re-enabled (was commented out in unit_tests.sh under ROCM-33746). phase and the
        # bitwise_and/or/xor cases below build their second operand with fn.rotate(angle=45)
        # (default bilinear interpolation), an open RPP defect unrelated to these kernels
        # themselves. Empirically re-confirmed locally on gfx1201: HOST fails for real, HIP
        # passes cleanly -- see the matching "_host"-suffixed entries in
        # test/common/skip_list.py and test/README.md "Validation" for the exact numbers. Only
        # the HOST verdict is skipped; HIP is verified for real.
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name phase --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Phase_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name threshold --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Threshold_${rgb_name[$rgb]}_${device_name}"
        # Re-enabled on both backends (unit_tests.sh only ever ran this on HIP, citing a known RPP
        # bug that blackens in-bounds output on HOST). The real, empirical re-run on gfx1201 shows
        # the OPPOSITE of that legacy assumption: HOST passes cleanly and HIP fails reproducibly.
        # The "WarpPerspective_*_hip" skip-list entry suppresses only the HIP verdict; HOST is
        # verified for real. See test/common/skip_list.py and test/README.md "Validation" -- this
        # discrepancy with the legacy comment is flagged there for a human to reconcile, not
        # silently resolved either way.
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name warp_perspective --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}WarpPerspective_${rgb_name[$rgb]}_${device_name}"

        # Re-enabled (was commented out in unit_tests.sh, "Disable until RPP fix is merged", no
        # ticket cited). Empirically re-confirmed locally on gfx1201: this op passes cleanly on
        # BOTH backends today. Not skip-listed -- verified for real on both backends.
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name remap --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Remap_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name crop_and_patch --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}CropAndPatch_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name ricap --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Ricap_${rgb_name[$rgb]}_${device_name}"
        # Re-enabled alongside phase above, same rotate-composition root cause (ROCM-33746);
        # HOST fails for real, HIP passes cleanly -- see skip_list.py.
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name bitwise_and --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}BitwiseAnd_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name bitwise_or --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}BitwiseOr_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name bitwise_xor --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}BitwiseXor_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name bitwise_not --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}BitwiseNot_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name erase --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Erase_${rgb_name[$rgb]}_${device_name}"

        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name gaussian_noise --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}GaussianNoise_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name shot_noise --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}ShotNoise_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name spatter --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Spatter_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name water --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Water_${rgb_name[$rgb]}_${device_name}"
        # ColorJitter is CPU-only, as unit_tests.cpp already assumes, and only a host golden exists.
        if [ $rgb -eq 1 ] && [ $device -eq 0 ]; then
            python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name color_jitter --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}ColorJitter_${rgb_name[$rgb]}_${device_name}"
        fi

        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name jpeg_compression_distortion --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}JpegCompressionDistortion_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name lut --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}LUT_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name posterize --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Posterize_${rgb_name[$rgb]}_${device_name}"
        python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name solarize --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}Solarize_${rgb_name[$rgb]}_${device_name}"
        if [ $rgb -eq 1 ]; then
            python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name channel_permute --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}ChannelPermute_${rgb_name[$rgb]}_${device_name}"
            python"$ver" "$unit_test_py" --image-dataset-path "$image_path" --augmentation-name color_to_greyscale --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}ColorToGreyscale_${rgb_name[$rgb]}_${device_name}"
        fi

        # Special Case - One Hot Encoded Labels
        python"$ver" "$unit_test_py" --image-dataset-path "$one_hot_data_path" --augmentation-name one_hot --batch-size $batch_size --max-width $width --max-height $height --color-format $rgb --$backend_arg -f "${output_path}OneHot_${rgb_name[$rgb]}_${device_name}"

    done
done

pwd

# Run the new, de-duplicated comparator (test/common/image_comparison.py) instead of the
# legacy tests/python_api/image_comparison.py.
python3 "$comparator" "$golden_output_path" "$output_path"
