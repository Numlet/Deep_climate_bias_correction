#!/bin/tcsh
#SBATCH --job-name=create_model
#SBATCH --output=job.out
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --account=pr04

python create_model_with_eager.py
