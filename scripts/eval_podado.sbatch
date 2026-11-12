#!/bin/bash
#SBATCH --job-name=eval_pwoa
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/mnist/eval_%j.out
#SBATCH --error=logs/mnist/eval_%j.err

source ~/pwoa_env/bin/activate
source env.sh
python eval_all.py --model-file assets/models/mnist_lenet3_pwoa_prune_0.75_irregular_mix_0_ree20.pt --skip-aa --out results/mnist_pwoa_podado.json
