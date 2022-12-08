#!/bin/bash -e
#SBATCH --job-name=jupyter
#SBATCH -p cpu
#SBATCH --ntasks=1 --cpus-per-task=12 --mem-per-cpu=8G
#SBATCH --time=1-0
#SBATCH --output=%x-%j-%N.out --error=%x-%j-%N.out

hostname=$(hostname)
hostip=$(hostname -i | sed 's/\./\\\./g')
export PYTHONUNBUFFERED=1
stdbuf -i0 -o0 -e0 jupyter lab --no-browser --ip=0.0.0.0 2>&1 | sed -u "s/$hostname/$hostip/g"