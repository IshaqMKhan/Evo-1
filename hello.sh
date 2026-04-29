#!/bin/bash
#SBATCH --job-name=hello_py
#SBATCH --partition=large
#SBATCH --time=00:05:00
#SBATCH --output=hello-%j.out

python hello.py
