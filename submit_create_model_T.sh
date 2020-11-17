#!/bin/tcsh
#SBATCH --job-name=create_model_T
#SBATCH --output=job_T.out
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --account=pr04

python create_model_T2M_and_Prec.py
