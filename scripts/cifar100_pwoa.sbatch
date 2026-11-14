#!/bin/bash
#SBATCH --job-name=pwoa_cifar100
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=cluster-node6
#SBATCH --output=logs/cifar100/slurm_pr%x_%j.out
#SBATCH --error=logs/cifar100/slurm_pr%x_%j.err

# Reproducao do PwoA no CIFAR-100 (WideResNet-34-10, professor LBGAT6).
# Hiperparametros de scripts/cifar100.sh dos autores:
#   50 epocas ADMM (admm-epochs=3, rho=0.01) + 100 epocas fine-tuning
#   SGD, lr=0.01, retrain_lr=0.005, cosine scheduler, bs=128
#   lx=5e-7 (0.0000005), ly=2e-6 (0.000002), tau=30, alpha(lambda)=100
#
# Uso:
#   sbatch --job-name=pwoa_0.875_0.2 scripts/cifar100_pwoa.sbatch 0.875 0.2
#   sbatch --job-name=pwoa_0.9375_0.2 scripts/cifar100_pwoa.sbatch 0.9375 0.2

set -euo pipefail

PR=${1:-0.875}
MIX=${2:-0.2}
PREMODEL=${3:-Cui2020Learnable_34_10_LBGAT6.pt}

cd "$SLURM_SUBMIT_DIR"
source ~/pwoa_env/bin/activate
source env.sh

mkdir -p logs/cifar100/wrn34-10 results assets/models assets/data/cifar100-data

if [ ! -f "assets/models/${PREMODEL}" ]; then
    echo "ERRO: Modelo pre-treinado assets/models/${PREMODEL} nao encontrado!"
    exit 1
fi

SAVE_NAME="prune_${PR}_irregular_teacher_LBGAT_student_LBGAT_mix_${MIX}"
MODEL_OUT="cifar100_wrn34-10_${SAVE_NAME}.pt"

echo "Iniciando PwoA no CIFAR-100 (WRN-34-10): pr=${PR}, mix_ratio=${MIX}, premodel=${PREMODEL}, node=$(hostname)"

python bin/run_hsicbt -cfg config/general-prune-hsicbt-cifar100.yaml -m wrn34-10 \
    -ep 50 --admm-epochs 3 --rho 0.01 -pr "${PR}" -st irregular -lr 0.01 -bs 128 \
    -reopt sgd --lr-scheduler cosine -ree 100 -relr 0.005 -relx 0.0000005 -rely 0.000002 \
    -mf "${MODEL_OUT}" \
    -lm "${PREMODEL}" \
    --distill --distill_loss kl --distill_model_path "assets/models/${PREMODEL}" --distill_temp 30 --distill_alpha 100 \
    -adv adv --mix_ratio "${MIX}" --device cuda:0 \
    > "logs/cifar100/wrn34-10/${SAVE_NAME}.out" 2>&1

echo "Treino concluido: assets/models/${MODEL_OUT}"

# Avaliacao completa (CW-50 + AutoAttack e CW-20)
TAG_PR=$(echo "${PR}" | tr -d '.')
TAG_MIX=$(echo "${MIX}" | tr -d '.')

python eval_all.py \
    --dataset cifar100 \
    --model wrn34-10 \
    --model-file "assets/models/${MODEL_OUT}" \
    --cw-steps 50 \
    --out "results/cifar100_wrn34-10_pwoa_${TAG_PR}_mix${TAG_MIX}_final.json"

python eval_all.py \
    --dataset cifar100 \
    --model wrn34-10 \
    --model-file "assets/models/${MODEL_OUT}" \
    --cw-steps 20 \
    --skip-aa \
    --out "results/cifar100_wrn34-10_pwoa_${TAG_PR}_mix${TAG_MIX}_cw20.json"

echo "Avaliacoes concluidas com sucesso!"
