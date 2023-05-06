#!/bin/bash
#SBATCH --job-name=git_experiments
#SBATCH --mem 49G
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
# python -m src.main_model.git_late_fusion --exp_name git_skill_fusion_full  --predict --questions_last 


# Run Pred with LM Loss only on Answers
python -m src.main_model.git_late_fusion --exp_name git_skill_fusion_partial --predict --questions_last 

# Run Pred with Full LM Loss
# python -m src.main_model.git_late_fusion --exp_name git_skill_fusion_full  --predict --questions_last 