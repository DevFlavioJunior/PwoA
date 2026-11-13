# Contents

* [Acknowledgement](#acknowledgement)
* [Citing This Paper](#citing-this-paper)
* [Abstract](#abstract)
* [Environment Setup](#environment-setup)
* [Running Framework](#running-framework)


## Acknowledgement 
This repository contains the source code for the `Pruning Adversarially Robust Neural Networks without Adversarial Examples` project developed by the Northeastern University's SPIRAL research group. This research was generously supported by the National Science Foundation (grant CCF-1937500). 


## Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> T. Jian*, Z. Wang*, Y. Wang, J. Dy, S. Ioannidis, "Pruning Adversarially Robust Neural Networks without Adversarial Examples", ICDM, 2022.

## Abstract
Adversarial pruning compresses models while preserving robustness. Current methods require access to adversarial examples during pruning. This significantly hampers training efficiency. Moreover, as new adversarial attacks and training methods develop at a rapid rate, adversarial pruning methods need to be modified accordingly to keep up. In this work, we propose a novel framework to prune a previously trained robust neural network while maintaining adversarial robustness, without further generating adversarial examples. We leverage concurrent self-distillation and pruning to preserve knowledge in the original model as well as regularizing the pruned model via the Hilbert-Schmidt Information Bottleneck. We comprehensively evaluate our proposed framework and show its superior performance in terms of both adversarial robustness and efficiency when pruning architectures trained on the MNIST, CIFAR-10, and CIFAR-100 datasets against five state-of-the-art attacks.

## Environment Setup
Please install the python dependencies and packages found below:
```bash
pytorch-1.6.0
torchvision-0.7.0
numpy-1.16.1
scipy-1.3.1
tqdm-4.33.0
yaml-0.1.7
torchattacks
```

Please setup environment in the project root directory using:
```bash
source env.sh
```

After intalling "torchattacks" package, we need to modify one place as follows to make sure our framework work. Please go to the installed package directory (`/.../torchattacks/attacks/`), modify `pgd.py` by finding the line `outputs = self.model(adv_images)`, and insert the following code after it:
```bash
if type(outputs) == tuple:
    outputs = outputs[0]
```

## Running Framework
You could produce the main results of Table 2, 3, 4, 5 & Figure 3 (PwoA) by this repository. We will release the full version of the repository soon. 

To setup adversarially robust pre-trained models for pruning, we consider five adversarially trained models provided by open-source state-of-the-art work, summarized in Table 1 in our paper. Please download those models from their original repository and saved in './assets/models/' under this repo.

To reproduce experiments that we have in the paper, one could run our batch script by the following instruction:
```bash
mnist.sh         # PwoA on MNIST
cifar10.sh       # PwoA on CIFAR-10
cifar100.sh      # PwoA on CIFAR-100
```

Please refer to [./bin/run_hbar](./run_hbar) for more usages. The arguments in the code are self-explanatory.

---

# Reprodução do PwoA no cluster Apuana (CIN0114 - 2026.2)

Equipe: Flávio Henrique, José Lucas, Gabriel Clemente e Wilton Sales

Projeto da disciplina Técnicas de Ataque e Detecção de Intrusão (CIn/UFPE).
Artigo: Jian et al., "Pruning Adversarially Robust Neural Networks without Adversarial Examples", ICDM 2022 (arXiv 2210.04311).
Este repositório é um fork de neu-spiral/PwoA com ajustes de compatibilidade, scripts de treino e avaliação e resultados.

## Ambiente usado

Python 3.10.12, torch 2.14.1 (CUDA 13), torchvision 0.29.2, torchattacks 3.5.1, numpy 2.2.6, scipy 1.15.3, pyyaml, tqdm. GPU NVIDIA A100 80GB (o artigo usa Tesla V100).

    python3 -m venv ~/pwoa_env && source ~/pwoa_env/bin/activate
    pip install torch torchvision numpy scipy pyyaml tqdm torchattacks
    source env.sh
    python patches/apply_torchattacks_patch.py   # get_logits aceita modelos que devolvem tupla

## Alterações no código dos autores

- source/hsicbt/utils/misc.py: to_categorical cria a matriz identidade no mesmo dispositivo dos rótulos (exigido pelo PyTorch 2.x).
- torchattacks (patches/): get_logits aceita saída em tupla, como a LeNet3 do projeto.
- source/hsicbt/math/admm.py: a variável de ambiente PWOA_PRUNE_ALL=1 faz a poda incluir também as camadas lineares (por padrão só pesos de 4 dimensões, isto é, convoluções, e camadas chamadas fc).
- eval_all.py (novo): mede acurácia natural, FGSM, PGD10, PGD20, CW (PGD com a perda do CW) e AutoAttack (versão standard), com eps 0.3 e passo 0.01 no MNIST (e eps 8/255, passo 2/255 no CIFAR-10).
- source/hsicbt/utils/io.py: remove também o prefixo `module.` ao carregar checkpoints, caso presente.

## Como reproduzir o MNIST (LeNet3, poda 4x)

1. Baixar o modelo pré-treinado (Model Zoo do HBaR, http://hdl.handle.net/2047/D20419656) para assets/models/ (arquivo usado: 1636127344-0616112-c6.pt).
2. Baixar o MNIST uma vez no nó de login (os nós de computação podem não ter internet).
3. Treinar e podar:

        sbatch scripts/mnist_pwoa.sbatch                       # só convoluções
        sbatch scripts/mnist_pwoa_var.sbatch 0.75 1 all        # poda uniforme (todas as camadas)
        sbatch scripts/mnist_pwoa_var.sbatch 0.875 0 conv      # 8x, só convoluções

4. Avaliar (nomes dos modelos em assets/models/):

        sbatch scripts/eval_final.sbatch assets/models/<modelo>.pt <nome_do_resultado>

Resultados em results/*.json. Os arquivos *_aa500.json usam só 500 imagens e servem apenas de teste de tempo.

## Resultados no MNIST (10.000 imagens de teste, uma execução por configuração)

| Modelo | Zeros no modelo | Natural | FGSM | PGD10 | PGD20 | CW (20 passos) | CW (50 passos) | AutoAttack |
|---|---|---|---|---|---|---|---|---|
| Original (sem poda) | 0,00% | 98,84 | 96,75 | 97,98 | 97,05 | 97,16 | 95,14 | 91,86 |
| 4x, só convoluções | 1,19% | 98,81 | 96,01 | 97,70 | 96,37 | 96,36 | 93,88 | 89,69 |
| 4x, todas as camadas | 74,97% | 98,82 | 96,29 | 97,68 | 96,35 | 96,30 | 93,89 | 88,78 |
| 8x, só convoluções | 1,39% | 98,82 | 95,18 | 97,43 | 95,68 | 95,65 | 91,98 | 87,06 |

Artigo (Tabela II, PwoA, LeNet): 4x = 98,66 / 95,89 / 97,35 / 96,16 / CW 96,15 / AA 90,00; 8x = 98,66 / 95,69 / 97,13 / 95,61 / CW 95,60 / AA 87,37. Modelo original (Tabela I): 98,66 / PGD20 96,44 / CW 95,10 / AA 91,57.

## Como reproduzir o CIFAR-10 (ResNet-18 TRADES, poda 4x e 8x)

1. Baixar o modelo pré-treinado ResNet-18 (HBaR + TRADES) do Model Zoo do HBaR (http://hdl.handle.net/2047/D20419657) para `assets/models/1636127673-3280377-2f.pt`.
2. Preparar o ambiente e baixar o CIFAR-10 no nó de login:

        bash scripts/setup_cifar10.sh

3. Treinar e podar (50 épocas ADMM + 100 épocas fine-tuning, SGD/cosine, lr 0.01/0.005, lx=2e-5, ly=1e-4, tau=30, lambda=10, mix_ratio=0):

        sbatch scripts/cifar10_pwoa_075_full.sbatch                            # poda 4x completa (75%) + avaliacao CW-50/AA e CW-20
        sbatch scripts/cifar10_pwoa.sbatch 1636127673-3280377-2f.pt 0.875 0   # poda 8x (87.5%)

4. Avaliar (eps=8/255, passo=2/255, 10.000 imagens de teste):

        sbatch scripts/eval_cifar10.sbatch assets/models/1636127673-3280377-2f.pt cifar10_resnet18_original_final 50
        sbatch scripts/eval_cifar10.sbatch assets/models/cifar10_resnet18_pwoa_075_full.pt cifar10_resnet18_pwoa_075_full_final 50
        sbatch scripts/eval_cifar10.sbatch assets/models/cifar10_resnet18_pwoa_prune_0.875_irregular_mix_0_ree100.pt cifar10_resnet18_pwoa_0875_final 50

## Resultados no CIFAR-10 (ResNet-18 TRADES, 10.000 imagens de teste)

| Modelo | Zeros no modelo | Natural | FGSM | PGD10 | PGD20 | CW (20 passos) | CW (50 passos) | AutoAttack | Observação |
|---|---|---|---|---|---|---|---|---|---|
| Original (sem poda) | 0,00% | 84,05 | 58,84 | 53,70 | 52,84 | 50,93 | 50,80 | 49,84 | `1636127673-3280377-2f.pt` |
| 4x preliminar (`pr=0.75`) | 74,90% | 84,36 | 57,61 | 51,03 | 49,72 | 48,04 | 47,84 | 46,65 | 50 ADMM + 37/100 ajuste fino (limite 4h no node6, 12/11/2026) |
| 4x completo (`pr=0.75`) | 74,90% | 84,37 | 57,64 | 51,06 | 49,64 | 47,92 | 47,66 | 46,60 | 50 ADMM + 100/100 ajuste fino (refeito completo em 13/11/2026) |
| 8x completo (`pr=0.875`) | 87,38% | 84,44 | 55,71 | 46,39 | 44,35 | 43,43 | 42,89 | 41,69 | 50 ADMM + 100/100 ajuste fino (7596,72 s no node9, 12/11/2026) |

Artigo (Tabela I, Original ResNet-18 TRADES): Natural 84,10 / FGSM 58,97 / PGD10 53,76 / PGD20 52,92 / CW 51,00 / AA 49,43. Artigo estendido (Tabela VIII, PwoA 4x, ResNet-18): Natural 86,07 / PGD10 49,61.

Nota sobre a execução 4x: a primeira execução do modelo 4x (`cifar10_resnet18_pwoa_075_final.json`, em 12/11/2026) caiu no `cluster-node6` (~3,7x mais lento que o `cluster-node9`) e atingiu o limite inicial de 4h na época 37 de 100 do ajuste fino (após concluir todas as 50 épocas de ADMM e o `hard_prune`). Esse resultado preliminar foi preservado, e uma nova execução completa de 150 épocas (50 ADMM + 100 ajuste fino) foi concluída em 13/11/2026 com limite de 12h e `--exclude=cluster-node6`, salvando separadamente em `cifar10_resnet18_pwoa_075_full.pt`, `results/cifar10_resnet18_pwoa_075_full_final.json` e `results/cifar10_resnet18_pwoa_075_full_cw20.json`.

## Diferenças em relação ao artigo e limitações

- O código poda só as convoluções da LeNet3 (fc1 e fc2 ficam densas); o artigo descreve taxa uniforme. A variante PWOA_PRUNE_ALL=1 faz a poda uniforme. Na ResNet-18, as convoluções concentram 99,91% dos parâmetros (11.164.352 de 11.173.962), então a poda padrão já atinge 74,90% (4x) e 87,38% (8x) de zeros no modelo inteiro.
- O script dos autores liga o termo HSIC só no fine-tuning, otimiza com Adam, usa lx e ly iguais à metade dos da Tabela IX e 100 épocas de fine-tuning no MNIST; usamos 20, como no artigo. No CIFAR-10 são usadas 50 épocas de ADMM e 100 de fine-tuning com SGD e escalonador cosseno.
- O artigo não informa os passos do CW nem a versão do AutoAttack. Reportamos CW com 20 e 50 passos (20 casa com a Tabela II e 50 com a Tabela I).
- O modelo salvo é o de maior robustez sob PGD no conjunto de teste (não há validação), o que é otimista.
- Uma execução por configuração; GPU e versões de bibliotecas diferentes das do artigo.

## Em andamento

CIFAR-100 e Fashion-MNIST (dataset adicional).
