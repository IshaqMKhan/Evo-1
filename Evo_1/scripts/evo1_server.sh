#!/bin/bash
#SBATCH --job-name=evo1_server
#SBATCH --output=/home/qxc857/Evo-1/Evo_1/logs/eval/server_%j.out
#SBATCH --error=/home/qxc857/Evo-1/Evo_1/logs/eval/server_%j.err
#SBATCH --partition=large
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200_1g.18gb:1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G

# -------------------------
# Environment setup
# -------------------------
module load tuni/miniforge3/24.9.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Evo1

cd /home/qxc857/Evo-1/Evo_1

# -------------------------
# Logging setup
# -------------------------
mkdir -p /home/qxc857/Evo-1/Evo_1/logs/eval

echo "[INFO] Starting EVO1 SERVER job: $SLURM_JOB_ID"
echo "[INFO] Node: $(hostname)"

# GPU snapshot before loading model
nvidia-smi > /home/qxc857/Evo-1/Evo_1/logs/eval/server_gpu_start_${SLURM_JOB_ID}.txt

# -------------------------
# Run server
# -------------------------
python scripts/Evo1_server.py 2>&1 | tee /home/qxc857/Evo-1/Evo_1/logs/eval/server_runtime_${SLURM_JOB_ID}.log