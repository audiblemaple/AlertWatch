#!/bin/bash

case $1 in
    hailo8|hailo8r|hailo8l|hailo15h|hailo15m|hailo15l|hailo10h)
        hw_arch=$1
    ;;
    *)
        echo -e "\e[31mBad parameter for hardware architecture, use options:e[0m"
        echo -e "\e[35m\thailo8\n\thailo8r\n\thailo8l\n\thailo15h\n\thailo15m\n\thailo15l\n\thailo10h\n\e[0m"
        echo -e "\e[35mexample usage: hailo compiler --hw-arch 'hardware architecture' --output-dir output 'input_file'\e[0m"
        exit 1
    ;;
esac

if [[ -z "$2" ]]; then
    echo -e "\e[31mInput model path is required\n\e[0m"
    echo -e "\e[35mexample usage: hailo compiler --hw-arch 'hardware architecture' --output-dir output 'input_file'\e[0m"
    exit 1
fi
input_file=$2

echo -e "\n\e[33mCompiling model\e[0m"
mkdir -p output
hailo compiler --hw-arch "$hw_arch" --output-dir output "$input_file"
rm *log

echo -e "\n\e[32mModel $input_file compiled successfully\n\e[0m"
