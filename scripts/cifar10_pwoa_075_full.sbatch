#!/bin/bash
#SBATCH --job-name=pwoa_075_full
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=cluster-node6
#SBATCH --output=logs/cifar10/full075_%j.out
#SBATCH --error=logs/cifar10/full075_%j.err

set -e
source ~/pwoa_env/bin/activate
source env.sh
mkdir -p logs/cifar10 results

# 1. Garantir os JSONs de CW-20 do original, 4x preliminar e 8x (rapido, sem AutoAttack)
if [ ! -f results/cifar10_resnet18_original_cw20.json ]; then
    python eval_all.py --dataset cifar10 --model resnet18 \
        --model-file assets/models/1636127673-3280377-2f.pt \
        --cw-steps 20 --skip-aa --out results/cifar10_resnet18_original_cw20.json
fi

if [ ! -f results/cifar10_resnet18_pwoa_075_cw20.json ]; then
    python eval_all.py --dataset cifar10 --model resnet18 \
        --model-file assets/models/cifar10_resnet18_pwoa_prune_0.75_irregular_mix_0_ree100.pt \
        --cw-steps 20 --skip-aa --out results/cifar10_resnet18_pwoa_075_cw20.json
fi

if [ ! -f results/cifar10_resnet18_pwoa_0875_cw20.json ]; then
    python eval_all.py --dataset cifar10 --model resnet18 \
        --model-file assets/models/cifar10_resnet18_pwoa_prune_0.875_irregular_mix_0_ree100.pt \
        --cw-steps 20 --skip-aa --out results/cifar10_resnet18_pwoa_0875_cw20.json
fi

# 2. Treinar o 4x completo (50 epocas ADMM + 100 epocas fine-tuning) com nome separado
preTrades=1636127673-3280377-2f.pt
pr=0.75
st=irregular
mix=0
lx=0.00002
ly=0.0001

run_hsicbt -cfg config/general-prune-hsicbt-cifar10.yaml -m resnet18 \
    -ep 50 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.01 -bs 128 \
    -op sgd -reopt sgd --lr-scheduler cosine -ree 100 -relr 0.005 -relx ${lx} -rely ${ly} \
    -mf cifar10_resnet18_pwoa_075_full.pt \
    -lm ${preTrades} \
    --distill --distill_loss kl --distill_model_path ./assets/models/${preTrades} --distill_temp 30 --distill_alpha 10 \
    -adv adv --mix_ratio ${mix} --device cuda:0 \
    > logs/cifar10/pwoa_075_full.out 2>&1

# 3. Avaliar o 4x completo com CW-50 + AutoAttack e com CW-20
python eval_all.py --dataset cifar10 --model resnet18 \
    --model-file assets/models/cifar10_resnet18_pwoa_075_full.pt \
    --cw-steps 50 --out results/cifar10_resnet18_pwoa_075_full_final.json

python eval_all.py --dataset cifar10 --model resnet18 \
    --model-file assets/models/cifar10_resnet18_pwoa_075_full.pt \
    --cw-steps 20 --skip-aa --out results/cifar10_resnet18_pwoa_075_full_cw20.json
