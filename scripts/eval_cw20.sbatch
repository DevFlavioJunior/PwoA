#!/bin/bash
#SBATCH --job-name=eval_cw20
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/mnist/evalcw20_%j.out
#SBATCH --error=logs/mnist/evalcw20_%j.err

source ~/pwoa_env/bin/activate
source env.sh
python eval_all.py --model-file "$1" --skip-aa --cw-steps 20 --out "results/$2.json"
