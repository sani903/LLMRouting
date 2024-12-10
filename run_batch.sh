#!/bin/sh
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=general
#SBATCH --mem=64GB
#SBATCH --time 11:00:00
#SBATCH --job-name=gen_openchat
#SBATCH --error=/home/ambuja/error/generation_openchat_anlp.err
#SBATCH --output=/home/ambuja/output/generation_openchat_anlp.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ambuja@andrew.cmu.edu

mkdir -p /scratch/ambuja/model
source ~/miniconda3/etc/profile.d/conda.sh

conda activate llm_routing

python vicuna_gsm8k_generation.py