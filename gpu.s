#!/bin/bash
#
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=myjob
#SBATCH --partition=secondary
#SBATCH --output=viewgpulog.log
#SBATCH ‑‑gres=gpu:V100:2
##SBATCH --error=myjob.e%j
##SBATCH --mail-user=yerong2@illinois.edu

#
# End of embedded SBATCH options
#

module load cuda/10.0
nvcc -V >> viewgpu.log
nvidia-smi >> vewgpu.log
