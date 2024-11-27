#!/bin/bash

case $1 in
    hailo8|hailo8r|hailo8l|hailo15h|hailo15m|hailo15l|hailo10h)
        hw_arch=$1
    ;;
    *)
        echo -e "\e[31mBad parameter for hardware architecture, use options:e[0m"
        echo -e "\e[35m\thailo8\n\thailo8r\n\thailo8l\n\thailo15h\n\thailo15m\n\thailo15l\n\thailo10h\n\e[0m"
        echo -e "\e[35mexample usage: hailo optimize --hw-arch 'hardware architecture' --calib-set-path 'calibration set' --output-har-path 'output file name' 'input file'\e[0m"
        exit 1
    ;;
esac

if [[ -z "$2" ]]; then
    echo -e "\e[31mCalibration set is required\n\e[0m"
    echo -e "\e[35mexample usage: hailo optimize --hw-arch 'hardware architecture' --calib-set-path 'calibration set' --output-har-path 'output file name' 'input file'\e[0m"
    exit 1
fi
calib_set=$2

if [ -z "$3" ]; then
    echo -e "\e[31mOutput model path is required\n\e[0m"
    echo -e "\e[35mexample usage: hailo optimize --hw-arch 'hardware architecture' --calib-set-path 'calibration set' --output-har-path 'output file name' 'input file'\e[0m"
    exit 1
fi
output_path=$3

if [ -z "$4" ]; then
    echo -e "\e[31mInput model path is required\n\e[0m"
    echo -e "\e[35mexample usage: hailo optimize --hw-arch 'hardware architecture' --calib-set-path 'calibration set' --output-har-path 'output file name' 'input file'\e[0m"
    exit 1
fi
input_file=$4

echo -e "\n\e[33mOptimizing model\e[0m"
mkdir "output"
hailo optimize --hw-arch "$hw_arch" --calib-set-path "$calib_set" --work-dir $(pwd) --output-har-path "$output_path" "$input_file"
rm *log

echo -e "\n\e[32mModel $input_file optimized successfully\n\e[0m"
