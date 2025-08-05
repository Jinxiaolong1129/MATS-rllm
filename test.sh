#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --account=dawn_song
#SBATCH --mail-type=all
#SBATCH --mail-user=jinxiaolong1129@gmail.com
#SBATCH --output=slurm_mats_setup_out.txt
#SBATCH --error=slurm_mats_setup_error.txt

echo "Test job running"