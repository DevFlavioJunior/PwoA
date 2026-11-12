#!/bin/bash
#SBATCH --job-name=pwoa_cifar10
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cifar10/slurm_%j.out
#SBATCH --error=logs/cifar10/slurm_%j.err

source ~/pwoa_env/bin/activate
source env.sh
mkdir -p logs/cifar10

# Argumentos opcionais:
# $1 = nome do arquivo do modelo professor dentro de assets/models/ (padrao: 1636127673-3280377-2f.pt)
# $2 = prune_ratio (padrao: 0.75 -> poda 4x)
# $3 = 1 para podar tambem a camada linear final (PWOA_PRUNE_ALL=1), 0 apenas convs (padrao: 0)
preTrades=${1:-1636127673-3280377-2f.pt}
pr=${2:-0.75}
if [ "${3:-0}" = "1" ]; then
    export PWOA_PRUNE_ALL=1
    tag="all"
else
    tag="prune"
fi

dataset=cifar10
model=resnet18
st=irregular
mix=0
lx=0.00002
ly=0.0001
save_name=pwoa_${tag}_${pr}_${st}_mix_${mix}_ree100

run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml -m ${model} \
    -ep 50 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.01 -bs 128 \
    -op sgd -reopt sgd --lr-scheduler cosine -ree 100 -relr 0.005 -relx ${lx} -rely ${ly} \
    -mf ${dataset}_${model}_${save_name}.pt \
    -lm ${preTrades} \
    --distill --distill_loss kl --distill_model_path ./assets/models/${preTrades} --distill_temp 30 --distill_alpha 10 \
    -adv adv --mix_ratio ${mix} --device cuda:0 \
    > logs/${dataset}/${save_name}.out 2>&1
