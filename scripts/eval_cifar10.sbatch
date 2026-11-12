#!/bin/bash
#SBATCH --job-name=eval_cifar10
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cifar10/eval_%j.out
#SBATCH --error=logs/cifar10/eval_%j.err

source ~/pwoa_env/bin/activate
source env.sh
mkdir -p logs/cifar10 results

# Uso:
#   sbatch scripts/eval_cifar10.sbatch [caminho_modelo] [nome_saida_sem_json] [cw_steps]
# Exemplo (modelo original):
#   sbatch scripts/eval_cifar10.sbatch assets/models/1636127673-3280377-2f.pt cifar10_resnet18_original_final 50
# Exemplo (modelo podado 4x):
#   sbatch scripts/eval_cifar10.sbatch assets/models/cifar10_resnet18_pwoa_prune_0.75_irregular_mix_0_ree100.pt cifar10_resnet18_pwoa_075_final 50

MODEL_FILE=${1:-assets/models/1636127673-3280377-2f.pt}
OUT_NAME=${2:-cifar10_resnet18_original_final}
CW_STEPS=${3:-50}

python eval_all.py \
    --dataset cifar10 \
    --model resnet18 \
    --model-file "$MODEL_FILE" \
    --cw-steps "$CW_STEPS" \
    --out "results/${OUT_NAME}.json"
