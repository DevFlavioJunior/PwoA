"""Aplica o ajuste no torchattacks: get_logits passa a aceitar modelos que devolvem tupla
(a LeNet do PwoA devolve (saida, ativacoes)). Rode com o ambiente virtual ativo."""
import os
import torchattacks

p = os.path.join(os.path.dirname(torchattacks.__file__), "attack.py")
# [EQUIPE] Le o arquivo preservando o tipo de quebra de linha (CRLF ou LF)
s = open(p, newline="").read()
eol = "\r\n" if "\r\n" in s else "\n"
old = eol.join(["        logits = self.model(inputs)", "        return logits", ""])
new = eol.join(["        logits = self.model(inputs)", "        if isinstance(logits, tuple):",
                "            logits = logits[0]", "        return logits", ""])
# Se o trecho novo ja existe, nada a fazer; senao troca o trecho antigo (que deve aparecer uma unica vez)
if new in s:
    print("ja aplicado")
else:
    assert s.count(old) == 1, "trecho nao encontrado exatamente uma vez"
    open(p, "w", newline="").write(s.replace(old, new))
    print("aplicado")
