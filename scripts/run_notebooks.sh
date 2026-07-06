#!/bin/bash
# ==============================================================================
# Script to run all polar_analysis Jupyter notebooks headlessly.
# This script executes each notebook in-place using jupyter nbconvert.
# ==============================================================================

set -e

NOTEBOOKS_DIR="jupyter"

echo "=================================================================="
echo "Starting Polar Analysis Diagnostics Workflow Execution"
echo "=================================================================="

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Error: 'jupyter' command not found. Please activate your conda environment."
    exit 1
fi

notebooks=(
    "Mean_Climate_Analysis.ipynb"
    "Variability_Modes_Analysis.ipynb"
    "ASL_Analysis.ipynb"
    "SAM_Analysis.ipynb"
    "Nino_Analysis.ipynb"
    "Regional_Analysis.ipynb"
    "EOF_Analysis.ipynb"
    "Trend_Analysis.ipynb"
    "Weather_Regimes_Analysis.ipynb"
)

for nb in "${notebooks[@]}"; do
    nb_path="${NOTEBOOKS_DIR}/${nb}"
    if [ -f "$nb_path" ]; then
        echo "--------------------------------------------------"
        echo "Executing: ${nb}..."
        echo "--------------------------------------------------"
        jupyter nbconvert --to notebook --execute --inplace "$nb_path"
    else
        echo "Warning: Notebook ${nb_path} not found, skipping."
    fi
done

echo "=================================================================="
echo "All notebooks executed successfully!"
echo "=================================================================="
