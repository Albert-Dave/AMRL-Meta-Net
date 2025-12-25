#!/bin/bash
set -e

echo "=============================="
echo " AMRL – Google Colab Run "
echo "=============================="

pip install -r requirements.txt

python colab_run.py
