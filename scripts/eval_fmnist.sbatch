#!/bin/bash
#SBATCH --job-name=eval_fmnist
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/fmnist/eval_%j.out
#SBATCH --error=logs/fmnist/eval_%j.err

source ~/pwoa_env/bin/activate
source env.sh
mkdir -p logs/fmnist

# Uso: sbatch scripts/eval_fmnist.sbatch assets/models/<arquivo_modelo.pt> <nome_saida_sem_json>
python eval_all.py --dataset fmnist --eps 0.1 --model-file "$1" --cw-steps 50 --out "results/$2.json"
