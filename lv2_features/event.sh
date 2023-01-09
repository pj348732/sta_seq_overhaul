#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/lv2_features/logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/lv2_features/logs/%A-%a-%x.out
#SBATCH --mem-per-cpu=32G --ntasks=1
#SBATCH --array=0-249
#SBATCH --cpus-per-task=4
#SBATCH -t 1-0
#SBATCH --partition=gpu


conda init bash
srun -l python lv2_norm.py