#!/bin/bash
#SBATCH --job-name=eval_aa
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/mnist/evalaa_%j.out
#SBATCH --error=logs/mnist/evalaa_%j.err

source ~/pwoa_env/bin/activate
source env.sh
python eval_all.py --model-file assets/models/1636127344-0616112-c6.pt --skip-aa --out results/mnist_original.json
python eval_all.py --model-file assets/models/1636127344-0616112-c6.pt --skip-aa --cw-steps 50 --out results/mnist_original_cw50.json
python eval_all.py --model-file assets/models/mnist_lenet3_pwoa_prune_0.75_irregular_mix_0_ree20.pt --n 500 --out results/mnist_pwoa_podado_aa500.json
