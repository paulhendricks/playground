#!/usr/bin/env bash
echo "Creating conda environment..."
conda env create -f environment.yml
conda env update

echo "Conda environment created! Make sure to run \`source activate playground\` whenever you open a new terminal and want to run programs under playground."
