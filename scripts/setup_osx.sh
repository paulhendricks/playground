#!/usr/bin/env bash
echo "Creating conda environment..."
conda env create --name playground python=3.5
source activate playground
easy_install pip
easy_install --upgrade six
export TF_BINARY_URL=export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0rc0-py3-none-any.whl
pip install --upgrade $TF_BINARY_URL

echo "Conda environment created! Make sure to run \`source activate playground\` whenever you open a new terminal and want to run programs under playground."
