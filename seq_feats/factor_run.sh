#!/bin/sh
#SBATCH --output=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/seq_feats/logs/%A-%a-%x.out
#SBATCH --error=/b/home/pengfei_ji/projects/pengfei_ji/sta_seq_overhaul/seq_feats/logs/%A-%a-%x.out
#SBATCH --mem-per-cpu=20G --ntasks=1
#SBATCH --array=0-249
#SBATCH --cpus-per-task=4
#SBATCH -t 1-0
#SBATCH --partition=gpu


conda init bash
srun -l python stav2_factors.py