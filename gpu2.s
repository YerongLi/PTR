#!/bin/bash
#
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=myjob
#SBATCH --partition=secondary
#SBATCH --output=R-%x.%j.out
#SBATCH --gres=gpu:A40:2
##SBATCH --error=myjob.e%j
##SBATCH --mail-user=yerong2@illinois.edu

#
# End of embedded SBATCH options
#

module load cuda/.11.1
nvcc -V >> viewgpu2.log
nvidia-smi >> vewgpu2.log
