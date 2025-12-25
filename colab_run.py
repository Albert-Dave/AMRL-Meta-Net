"""
One-click Google Colab runner for AMRL.

Usage (Colab):
!python colab_run.py
"""

import os
import subprocess
import sys

# --------------------------------------------------
# 1. Install dependencies (Colab-safe)
# --------------------------------------------------
def install_requirements():
    if "google.colab" in sys.modules:
        print("Running in Google Colab — installing dependencies...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )

install_requirements()

# --------------------------------------------------
# 2. Imports after installation
# --------------------------------------------------
from utils.seed import set_seed
from train_meta import main as train_meta_main
from evaluate import main as evaluate_main

# --------------------------------------------------
# 3. 
# --------------------------------------------------
set_seed(42)

# --------------------------------------------------
# 4. Run Meta-Training (Multi-task AMRL)
# --------------------------------------------------
print("\n==============================")
print(" Starting AMRL Meta-Training ")
print("==============================\n")

train_meta_main()

# --------------------------------------------------
# 5. Run Evaluation
# --------------------------------------------------
print("\n==============================")
print(" Running Evaluation ")
print("==============================\n")

evaluate_main()

print("\n==============================")
print(" AMRL pipeline completed ")
print("==============================\n")
