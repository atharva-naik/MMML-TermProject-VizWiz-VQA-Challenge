#!/bin/bash
#SBATCH --job-name=git_experiments
#SBATCH --mem 50G
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=tir-0-32
#SBATCH --ntasks 1
#SBATCH --output /projects/tir5/users/nvaikunt/MMML-TermProject-VizWiz-VQA-Challenge/slurm_logs/log-%x-%J.txt

#Load conda environment
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate vizwiz
cd /projects/tir5/users/nvaikunt/MMML-TermProject-VizWiz-VQA-Challenge

# Run Git with Full LM Loss
python -m src.main_model.git_late_fusion --exp_name git_skill_fusion_full --train  --epochs 5 --questions_last 

# Run Git with LM Loss only on Answers
python -m src.main_model.git_late_fusion --exp_name git_skill_fusion_partial --train  --epochs 5 --questions_last --partial_loss
