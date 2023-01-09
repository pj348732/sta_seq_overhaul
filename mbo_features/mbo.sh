#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/mbo_features/logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/mbo_features/logs/%A-%a-%x.out
#SBATCH --mem-per-cpu=16G --ntasks=1
#SBATCH --array=0-249
#SBATCH --cpus-per-task=4
#SBATCH -t 1-0
#SBATCH --partition=cpu


conda init bash
srun -l python mbo_label.py