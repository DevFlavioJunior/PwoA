#!/bin/bash
set -e

source ~/pwoa_env/bin/activate
source env.sh

mkdir -p logs/cifar10 assets/models assets/data/cifar10-data results

echo "[1/2] Aplicando patch no torchattacks..."
python patches/apply_torchattacks_patch.py

echo "[2/2] Baixando CIFAR-10 no no de login (os nos de computacao nao tem internet)..."
python -c "from torchvision import datasets; datasets.CIFAR10('./assets/data/cifar10-data', train=True, download=True); datasets.CIFAR10('./assets/data/cifar10-data', train=False, download=True)"

echo "Pronto! Coloque o modelo pre-treinado do HBaR em assets/models/ e rode o eval_cifar10.sbatch."
