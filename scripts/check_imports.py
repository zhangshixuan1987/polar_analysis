#!/usr/bin/env python3
# ==============================================================================
# Script to verify that all modules in the refactored 'util' package
# import correctly without errors.
# ==============================================================================

import sys
import os

# Append the parent directory to sys.path so we can import 'util'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

modules_to_test = [
    "util.common",
    "util.asl",
    "util.eof",
    "util.mean_climate",
    "util.mov_analysis",
    "util.nino",
    "util.regional",
    "util.sam",
    "util.trend",
    "util.weather_regimes"
]

print("==================================================================")
# Print current python executable and version
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")
print("Starting Import Verification for 'util' package...")
print("==================================================================")

failures = 0
for mod in modules_to_test:
    try:
        print(f"Importing {mod}... ", end="")
        __import__(mod)
        print("SUCCESS")
    except Exception as e:
        print("FAILED")
        print(f"Error details:\n{e}\n")
        failures += 1

print("==================================================================")
if failures == 0:
    print("ALL MODULES IMPORTED SUCCESSFULLY!")
    sys.exit(0)
else:
    print(f"FAILED: {failures} modules failed to import.")
    sys.exit(1)
