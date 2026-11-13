# Verifica se um modelo podado tem ESTRUTURA: para cada camada de pesos, mostra a fracao de
# pesos zero, de linhas inteiras zeradas (filtros/neuronios de saida) e de colunas inteiras zeradas
# (entradas: in_canal x posicao do kernel nas convolucoes; neuronios de entrada nas camadas lineares).
# Uso: python check_estrutura.py assets/models/<modelo>.pt
import sys
import torch

sd = torch.load(sys.argv[1], map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
if not isinstance(sd, dict):          # modelo salvo inteiro
    sd = sd.state_dict()

tot_z = tot_n = 0
print("%-14s %-18s %8s %10s %10s" % ("camada", "forma", "zeros%", "linhas0%", "colunas0%"))
for name, w in sd.items():
    if "weight" not in name or w.dim() not in (2, 4):
        continue
    w2 = w.reshape(w.shape[0], -1)               # linhas = saidas, colunas = entradas
    z = (w == 0).sum().item()
    lin0 = ((w2 == 0).all(dim=1)).float().mean().item() * 100
    col0 = ((w2 == 0).all(dim=0)).float().mean().item() * 100
    tot_z += z; tot_n += w.numel()
    print("%-14s %-18s %8.2f %10.2f %10.2f" % (name, tuple(w.shape), 100 * z / w.numel(), lin0, col0))
print("TOTAL pesos: %d, zeros: %d (%.2f%%)" % (tot_n, tot_z, 100 * tot_z / tot_n))
