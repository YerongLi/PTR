#!/bin/bash
#
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=myjob
#SBATCH --partition=secondary
#SBATCH --output=ptrlog.log
#SBATCH --gres=gpu:V100:1
##SBATCH --error=myjob.e%j
##SBATCH --mail-user=yerong2@illinois.edu

#
# End of embedded SBATCH options
#

module load cuda/.11.6
conda activate /scratch/yerong/.conda/envs/ptr
bash scripts/run_large_retacred.sh > ptr.log
