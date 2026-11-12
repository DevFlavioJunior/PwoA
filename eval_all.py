"""Avaliacao completa de robustez de um modelo (LeNet3, ResNet-18, WRN) no MNIST, Fashion-MNIST, CIFAR-10 ou CIFAR-100.

Escrito pela equipe (nao faz parte do codigo original dos autores).
Mede, no conjunto de teste, a acuracia natural e sob ataques adversariais
(FGSM, PGD10, PGD20, CW e AutoAttack), alem da fracao de pesos zerados.
A versao com suporte ao CIFAR (modelos, raio, carregamento do checkpoint) e do Gabriel Clemente.

Raio e passo padrao por conjunto de dados: MNIST 0.3 / 0.01; Fashion-MNIST 0.1 / 0.01
(escolha nossa); CIFAR 8/255 / 2/255 (como no artigo).

Exemplo:
    python eval_all.py --model-file assets/models/<modelo>.pt --dataset cifar10 --cw-steps 50 --out results/x.json
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
import torchattacks
from torchvision import datasets, transforms
from hsicbt.model.lenet import LeNet3
from hsicbt.model.resnet import ResNet18, ResNet50
from hsicbt.model.wideresnet import WideResNet28_10, WideResNet34_10

# ---------------------------------------------------------------- argumentos
p = argparse.ArgumentParser()
p.add_argument("--model-file", required=True)
p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "cifar10", "cifar100"])
p.add_argument("--model", default="", help="lenet3, resnet18, wrn28-10, wrn34-10 (auto-detecta pelo dataset se vazio)")
# quantas imagens de teste usar (10000 = conjunto de teste completo)
p.add_argument("--n", type=int, default=10000)
p.add_argument("--bs", type=int, default=500)
p.add_argument("--eps", type=float, default=None)
p.add_argument("--alpha", type=float, default=None)
# passos do CW (o artigo nao informa; usamos 20 e 50)
p.add_argument("--cw-steps", type=int, default=20)
# pula o AutoAttack (e o ataque mais lento)
p.add_argument("--skip-aa", action="store_true")
p.add_argument("--device", default="cuda:0")
p.add_argument("--out", default="")
a = p.parse_args()

# Modelo padrao por conjunto de dados: ResNet-18 no CIFAR, LeNet3 nos demais
if not a.model:
    a.model = "resnet18" if a.dataset in ("cifar10", "cifar100") else "lenet3"

# Raio da bola l-infinito: 0.3 (MNIST), 0.1 (Fashion-MNIST, escolha nossa), 8/255 (CIFAR, como no artigo)
if a.eps is None:
    if a.dataset in ("cifar10", "cifar100"):
        a.eps = 8.0 / 255.0
    elif a.dataset == "fmnist":
        a.eps = 0.1
    else:
        a.eps = 0.3

# Passo dos ataques iterativos: 2/255 no CIFAR e 0.01 nos demais
if a.alpha is None:
    if a.dataset in ("cifar10", "cifar100"):
        a.alpha = 2.0 / 255.0
    else:
        a.alpha = 0.01

# Sementes fixas: os ataques com inicio aleatorio ficam reproduziveis
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
dev = a.device
# Numero de classes (usado no CW e no AutoAttack)
num_classes = 100 if a.dataset == "cifar100" else 10

# ---------------------------------------------------------------- modelo
if a.model == "lenet3":
    model = LeNet3(data_code=a.dataset, robustness=True)
elif a.model == "resnet18":
    model = ResNet18(num_classes=num_classes, robustness=True)
elif a.model == "resnet50":
    model = ResNet50(num_classes=num_classes, robustness=True)
elif a.model in ("wideresnet", "wrn28-10"):
    model = WideResNet28_10(num_classes=num_classes, robustness=True)
elif a.model == "wrn34-10":
    model = WideResNet34_10(num_classes=num_classes, robustness=True)
else:
    raise ValueError(f"Modelo desconhecido: {a.model}")

# Carrega o checkpoint e limpa os nomes das chaves (prefixos "module." / "module.model.")
sd = torch.load(a.model_file, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]

own_state = model.state_dict()
cleaned_sd = {}
for name, param in sd.items():
    clean_name = name.replace("module.model.", "")
    if clean_name not in own_state and clean_name.startswith("module."):
        clean_name = clean_name[len("module."):]
    if clean_name not in own_state:
        print("not found: ", name, flush=True)
        continue
    cleaned_sd[clean_name] = param

for k in own_state.keys():
    if k not in cleaned_sd:
        print("missing in checkpoint: ", k, flush=True)

# strict=True: se alguma chave nao casar, o script para com erro (antes era strict=False)
model.load_state_dict(cleaned_sd, strict=True)
model.to(dev).eval()

# ---------------------------------------------------------------- dados (so o conjunto de teste, sem normalizacao: ToTensor deixa os pixels em [0,1])
if a.dataset == "mnist":
    test = datasets.MNIST("./assets/data/mnist-data", train=False, download=False,
                          transform=transforms.ToTensor())
elif a.dataset == "fmnist":
    test = datasets.FashionMNIST("./assets/data/fasionmnist-data", train=False, download=False,
                                 transform=transforms.ToTensor())
elif a.dataset == "cifar10":
    test = datasets.CIFAR10("./assets/data/cifar10-data", train=False, download=False,
                            transform=transforms.ToTensor())
elif a.dataset == "cifar100":
    test = datasets.CIFAR100("./assets/data/cifar100-data", train=False, download=False,
                             transform=transforms.ToTensor())

n = min(a.n, len(test))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test, range(n)),
                                     batch_size=a.bs, shuffle=False)

# Esparsidade: fracao de parametros iguais a zero no modelo INTEIRO
zeros = sum((p_ == 0).sum().item() for p_ in model.parameters())
total = sum(p_.numel() for p_ in model.parameters())
print(f"modelo: {a.model_file} ({a.model} em {a.dataset}, eps={a.eps:.6f}, alpha={a.alpha:.6f})")
print(f"zeros: {zeros} de {total} parametros ({100.0*zeros/total:.2f}%)", flush=True)

# ---------------------------------------------------------------- ataque CW
def cw_pgd(x, y):
    # PGD com a funcao de perda do CW (margem entre a classe errada mais forte e a correta)
    # Inicio aleatorio dentro da bola, como no PGD
    x_adv = (x + torch.empty_like(x).uniform_(-a.eps, a.eps)).clamp(0, 1)
    onehot = F.one_hot(y, num_classes).bool()
    for _ in range(a.cw_steps):
        x_adv.requires_grad_(True)
        z = model(x_adv)
        # logit da classe correta
        correct = z[onehot]
        # maior logit entre as classes erradas
        other = z.masked_fill(onehot, -1e4).max(1)[0]
        grad = torch.autograd.grad((other - correct).sum(), x_adv)[0]
        # sobe a margem (aproxima do erro)
        x_adv = x_adv.detach() + a.alpha * grad.sign()
        # projeta na bola e em [0,1]
        x_adv = torch.min(torch.max(x_adv, x - a.eps), x + a.eps).clamp(0, 1)
    return x_adv.detach()

# ---------------------------------------------------------------- ataques da biblioteca torchattacks
fgsm = torchattacks.FGSM(model, eps=a.eps)
pgd10 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=10, random_start=True)
pgd20 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=20, random_start=True)

# Registro dos parametros usados (vai para o .json)
results = {
    "model_file": a.model_file,
    "dataset": a.dataset,
    "model": a.model,
    "eps": a.eps,
    "alpha": a.alpha,
    "cw_steps": a.cw_steps,
    "n": n,
    "zeros_pct": 100.0 * zeros / total,
}

# Gera os exemplos com make_adv(x, y), mede a acuracia e guarda em results[name].
def run(name, make_adv):
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
run("natural", lambda x, y: x)
run("FGSM", lambda x, y: fgsm(x, y))
run("PGD10", lambda x, y: pgd10(x, y))
run("PGD20", lambda x, y: pgd20(x, y))
run("CW", cw_pgd)
if not a.skip_aa:
    # AutoAttack versao "standard": ataque mais forte e mais lento
    aa = torchattacks.AutoAttack(model, norm="Linf", eps=a.eps, version="standard",
                                 n_classes=num_classes, seed=0)
    run("AA", lambda x, y: aa(x, y))

if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("salvo em", a.out)
