"""Avaliacao completa de robustez de um modelo LeNet3 (MNIST ou Fashion-MNIST).

Escrito pela equipe (nao faz parte do codigo original dos autores).
Mede, no conjunto de teste, a acuracia natural e sob ataques adversariais
(FGSM, PGD10, PGD20, CW e AutoAttack), alem da fracao de pesos zerados.

Exemplo:
    python eval_all.py --model-file assets/models/<modelo>.pt --cw-steps 50 --out results/x.json
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
import torchattacks
from torchvision import datasets, transforms
from hsicbt.model.lenet import LeNet3

# ---------------------------------------------------------------- argumentos
p = argparse.ArgumentParser()
p.add_argument("--model-file", required=True)           # checkpoint .pt a avaliar
p.add_argument("--n", type=int, default=10000)           # quantas imagens de teste usar
p.add_argument("--bs", type=int, default=500)            # tamanho do lote
p.add_argument("--eps", type=float, default=0.3)         # raio da bola l-infinito (0.3 no MNIST; 0.1 no nosso Fashion-MNIST)
p.add_argument("--alpha", type=float, default=0.01)      # passo dos ataques iterativos
p.add_argument("--cw-steps", type=int, default=20)       # passos do CW (o artigo nao informa; usamos 20 e 50)
p.add_argument("--skip-aa", action="store_true")         # pula o AutoAttack (e o mais lento)
p.add_argument("--device", default="cuda:0")
p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist"])
p.add_argument("--out", default="")                      # arquivo .json para salvar os resultados
a = p.parse_args()

# Sementes fixas: os ataques com inicio aleatorio ficam reproduziveis
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
dev = a.device

# ---------------------------------------------------------------- modelo
model = LeNet3(data_code=a.dataset, robustness=True)
sd = torch.load(a.model_file, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:          # alguns checkpoints guardam os pesos em "state_dict"
    sd = sd["state_dict"]
model.load_state_dict(sd)
model.to(dev).eval()

# ---------------------------------------------------------------- dados (so o conjunto de teste)
if a.dataset == "mnist":
    test = datasets.MNIST("./assets/data/mnist-data", train=False, download=False, transform=transforms.ToTensor())
else:
    test = datasets.FashionMNIST("./assets/data/fasionmnist-data", train=False, download=False, transform=transforms.ToTensor())
n = min(a.n, len(test))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test, range(n)),
                                     batch_size=a.bs, shuffle=False)

# ---------------------------------------------------------------- esparsidade
# Fracao de parametros iguais a zero no modelo INTEIRO (nao so nas camadas podadas)
zeros = sum((p_ == 0).sum().item() for p_ in model.parameters())
total = sum(p_.numel() for p_ in model.parameters())
print(f"modelo: {a.model_file}")
print(f"zeros: {zeros} de {total} parametros ({100.0*zeros/total:.2f}%)", flush=True)

# ---------------------------------------------------------------- ataque CW
def cw_pgd(x, y):
    # PGD com a funcao de perda do CW (margem entre a classe errada mais forte e a correta)
    # Inicio aleatorio dentro da bola, como no PGD
    x_adv = (x + torch.empty_like(x).uniform_(-a.eps, a.eps)).clamp(0, 1)
    onehot = F.one_hot(y, 10).bool()
    for _ in range(a.cw_steps):
        x_adv.requires_grad_(True)
        z = model(x_adv)                                  # logits (a LeNet3 devolve so a saida neste modo)
        correct = z[onehot]                               # logit da classe correta
        other = z.masked_fill(onehot, -1e4).max(1)[0]     # maior logit entre as classes erradas
        grad = torch.autograd.grad((other - correct).sum(), x_adv)[0]
        x_adv = x_adv.detach() + a.alpha * grad.sign()    # sobe a margem (aproxima do erro)
        x_adv = torch.min(torch.max(x_adv, x - a.eps), x + a.eps).clamp(0, 1)  # projeta na bola e em [0,1]
    return x_adv.detach()

# ---------------------------------------------------------------- ataques da biblioteca torchattacks
fgsm = torchattacks.FGSM(model, eps=a.eps)
pgd10 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=10, random_start=True)
pgd20 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=20, random_start=True)

results = {"model_file": a.model_file, "n": n, "zeros_pct": 100.0 * zeros / total}

def run(name, make_adv):
    """Gera os exemplos com make_adv(x, y), mede a acuracia e guarda em results[name]."""
    t0 = time.time()
    ok = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        xa = make_adv(x, y)
        with torch.no_grad():
            ok += (model(xa).argmax(1) == y).sum().item()
    acc = 100.0 * ok / n
    results[name] = acc
    print(f"{name:8s} {acc:6.2f}%   ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- execucao
run("natural", lambda x, y: x)                            # sem ataque
run("FGSM", lambda x, y: fgsm(x, y))
run("PGD10", lambda x, y: pgd10(x, y))
run("PGD20", lambda x, y: pgd20(x, y))
run("CW", cw_pgd)
if not a.skip_aa:
    # AutoAttack versao "standard": ataque mais forte, o mais lento (~23 min no MNIST)
    aa = torchattacks.AutoAttack(model, norm="Linf", eps=a.eps, version="standard",
                                 n_classes=10, seed=0)
    run("AA", lambda x, y: aa(x, y))

if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("salvo em", a.out)
