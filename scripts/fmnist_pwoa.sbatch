#!/bin/bash
#SBATCH --job-name=pwoa_fmnist
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/fmnist/slurm_%j.out
#SBATCH --error=logs/fmnist/slurm_%j.err

source ~/pwoa_env/bin/activate
source env.sh
mkdir -p logs/fmnist

# Argumentos opcionais:
# $1 = arquivo do modelo professor em assets/models/ (padrao: fmnist_lenet3_teacher.pt)
# $2 = prune_ratio (padrao: 0.75)
# $3 = 1 para podar todas as camadas (PWOA_PRUNE_ALL=1), 0 apenas conv (padrao: 1)
# $4 = tag do experimento (padrao: all ou conv conforme $3)
prePGD=${1:-fmnist_lenet3_teacher.pt}
pr=${2:-0.75}
prune_all=${3:-1}
if [ "$prune_all" = "1" ]; then
    export PWOA_PRUNE_ALL=1
    default_tag="all"
else
    default_tag="conv"
fi
tag=${4:-$default_tag}

dataset=fmnist
model=lenet3
st=irregular
mix=0
lx=0.0002
ly=0.00005
save_name=pwoa_${tag}_${pr}_${st}_mix_${mix}_ree20

run_hsicbt -cfg config/general-prune-hsicbt-${dataset}.yaml \
    -ep 50 --admm-epochs 3 --rho 0.01 -pr ${pr} -st $st -lr 0.0005 -bs 128 \
    -reopt adam --lr-scheduler default -ree 20 -relr 0.001 -relx ${lx} -rely ${ly} \
    -mf ${dataset}_${model}_${save_name}.pt \
    -lm ${prePGD} \
    --distill --distill_loss kl --distill_model_path ./assets/models/${prePGD} --distill_temp 30 --distill_alpha 10 \
    -adv adv --mix_ratio ${mix} --device cuda:0 \
    > logs/${dataset}/${save_name}.out 2>&1
