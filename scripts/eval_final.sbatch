#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/mnist/evalfinal_%j.out
#SBATCH --error=logs/mnist/evalfinal_%j.err

source ~/pwoa_env/bin/activate
source env.sh
python eval_all.py --model-file "$1" --cw-steps 50 --out "results/$2.json"
