#!/bin/bash

echo -e "\n\e[33mParsing model\e[0m"

hailo parser onnx $1
mv *.har output/
rm *log

echo -e "\n\e[32mModel $1 parsed successfully\e[0m"
