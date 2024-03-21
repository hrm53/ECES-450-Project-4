#!/bin/sh
#SBATCH --partition=bm
#SBATCH --mem=400G
#SBATCH --nodes=1      # nodes requested
#SBATCH --ntasks=24     # tasks requested
#SBATCH --output=./slurm/umap_%A_%a.txt 
#SBATCH --time=05:5:00

module load python/anaconda3
. ~/.bashrc



python -u umap_runner.py  
