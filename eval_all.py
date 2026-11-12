import argparse, json, os, time
import torch
import torch.nn.functional as F
import torchattacks
from torchvision import datasets, transforms
from hsicbt.model.lenet import LeNet3

p = argparse.ArgumentParser()
p.add_argument("--model-file", required=True)
p.add_argument("--n", type=int, default=10000)
p.add_argument("--bs", type=int, default=500)
p.add_argument("--eps", type=float, default=0.3)
p.add_argument("--alpha", type=float, default=0.01)
p.add_argument("--cw-steps", type=int, default=20)
p.add_argument("--skip-aa", action="store_true")
p.add_argument("--device", default="cuda:0")
p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist"])
p.add_argument("--out", default="")
a = p.parse_args()

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
dev = a.device

model = LeNet3(data_code=a.dataset, robustness=True)
sd = torch.load(a.model_file, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
model.load_state_dict(sd)
model.to(dev).eval()

if a.dataset == "mnist":
    test = datasets.MNIST("./assets/data/mnist-data", train=False, download=False, transform=transforms.ToTensor())
else:
    test = datasets.FashionMNIST("./assets/data/fasionmnist-data", train=False, download=False, transform=transforms.ToTensor())
n = min(a.n, len(test))
loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test, range(n)),
                                     batch_size=a.bs, shuffle=False)

zeros = sum((p_ == 0).sum().item() for p_ in model.parameters())
total = sum(p_.numel() for p_ in model.parameters())
print(f"modelo: {a.model_file}")
print(f"zeros: {zeros} de {total} parametros ({100.0*zeros/total:.2f}%)", flush=True)

def cw_pgd(x, y):
    # PGD com a funcao de perda do CW (margem entre a classe errada mais forte e a correta)
    x_adv = (x + torch.empty_like(x).uniform_(-a.eps, a.eps)).clamp(0, 1)
    onehot = F.one_hot(y, 10).bool()
    for _ in range(a.cw_steps):
        x_adv.requires_grad_(True)
        z = model(x_adv)
        correct = z[onehot]
        other = z.masked_fill(onehot, -1e4).max(1)[0]
        grad = torch.autograd.grad((other - correct).sum(), x_adv)[0]
        x_adv = x_adv.detach() + a.alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - a.eps), x + a.eps).clamp(0, 1)
    return x_adv.detach()

fgsm = torchattacks.FGSM(model, eps=a.eps)
pgd10 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=10, random_start=True)
pgd20 = torchattacks.PGD(model, eps=a.eps, alpha=a.alpha, steps=20, random_start=True)

results = {"model_file": a.model_file, "n": n, "zeros_pct": 100.0 * zeros / total}

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

run("natural", lambda x, y: x)
run("FGSM", lambda x, y: fgsm(x, y))
run("PGD10", lambda x, y: pgd10(x, y))
run("PGD20", lambda x, y: pgd20(x, y))
run("CW", cw_pgd)
if not a.skip_aa:
    aa = torchattacks.AutoAttack(model, norm="Linf", eps=a.eps, version="standard",
                                 n_classes=10, seed=0)
    run("AA", lambda x, y: aa(x, y))

if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=2)
    print("salvo em", a.out)
