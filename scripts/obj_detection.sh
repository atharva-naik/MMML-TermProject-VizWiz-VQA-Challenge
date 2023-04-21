#!/bin/bash
#SBATCH --job-name=deta-obj-detection
#SBATCH --mem 64G
#SBATCH --time=22:00:00
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=tir-0-32
#SBATCH --ntasks 1
#SBATCH --output /projects/tir5/users/nvaikunt/MMML-TermProject-VizWiz-VQA-Challenge/slurm_logs/log-%x-%J.txt

# Load conda environment
source /home/nvaikunt/miniconda3/etc/profile.d/conda.sh
conda activate vizwiz
cd /projects/tir5/users/nvaikunt/MMML-TermProject-VizWiz-VQA-Challenge

# Run Obj Detection Script
# For val data
# python -m src.object_detection.deta_detector 
# For train data
python -m src.object_detection.deta_detector --split "train" --image_dir "data/VQA/train" --save_dir ""data/VQA/train_obj_detect/coco" --results_dir "data/VQA/" --batch_sz 4