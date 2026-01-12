#!/bin/bash
# 示例运行脚本
# This script is designed to be run from any directory.

# Get the directory of this script to locate the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The project root is one level up from the script's directory.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Change to this script's directory, so all outputs will be saved here.
cd "${SCRIPT_DIR}"

# Add project root to PYTHONPATH to ensure the 'optimizer' module can be found.
# Convert path to Windows format if we are in a MINGW/MSYS environment
if [[ "$(uname -o)" == "Msys" || "$(uname -o)" == "Cygwin" ]]; then
    PROJECT_ROOT_NATIVE="$(cd "$PROJECT_ROOT" && pwd -W)"
    export PYTHONPATH="${PROJECT_ROOT_NATIVE};${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi


# Now run the main module.
# The python interpreter will find the 'optimizer' package via PYTHONPATH.
# Parameters are now loaded from optimizer/configs/config.yaml
python -m optimizer.core.main
