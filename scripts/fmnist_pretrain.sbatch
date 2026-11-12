#!/bin/bash
#SBATCH --job-name=fm_pre
#SBATCH --partition=short-simple
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/CIN/fhmj/projeto/PwoA
#SBATCH --output=logs/fmnist/slurm_%j.out
#SBATCH --error=logs/fmnist/slurm_%j.err

source ~/pwoa_env/bin/activate
source env.sh
ep=$1
name=$2
run_hsicbt -cfg config/pretrain-pgd-fmnist.yaml -ep ${ep} -mf fmnist_lenet3_${name}.pt \
    -adv adv --mix_ratio 1 --device cuda:0 \
    > logs/fmnist/pretrain_${name}.out 2>&1
