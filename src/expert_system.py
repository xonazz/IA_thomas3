"""Sistema especialista: base de conhecimento (arquivo .txt) separada do motor.

Formato de cada linha da base:
  ID | tipo | prioridade | SE <cond> [E <cond> ...] ENTÃO <conclusão>
Uma regra usa só conjunções (E) ou só disjunções (OU), nunca as duas.
Motor: encadeamento para frente com resolução de conflito por prioridade
(dispara-se a primeira regra aplicável; a regra final SE VERDADEIRO é o padrão).
"""
import json
import operator
import re
from dataclasses import dataclass

import numpy as np

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
       "<": operator.lt, "==": operator.eq}
COND_RE = re.compile(r"^\s*(\w+)\s*(>=|<=|==|>|<)\s*([-\w.]+)\s*$")


@dataclass
class Rule:
    id: str
    tipo: str
    prioridade: int
    conds: list          # [(attr, op, valor)]
    conector: str        # "E" ou "OU"
    conclusao: object    # str (classe) ou float (valor)

    def aplica(self, fatos):
        if not self.conds:
            return True
        res = [OPS[op](fatos[a], v) if not _ausente(fatos[a]) else False for a, op, v in self.conds]
        return all(res) if self.conector == "E" else any(res)


def _ausente(x):
    return isinstance(x, float) and x != x      # NaN


def _val(s):
    """Número se possível; senão texto (ex.: Yes, Female)."""
    try:
        return float(s)
    except ValueError:
        return s


def load_rules(path):
    regras = []
    for linha in open(path, encoding="utf-8"):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        rid, tipo, prio, corpo = [x.strip() for x in linha.split("|", 3)]
        m = re.match(r"^SE\s+(.*?)\s+ENTÃO\s+(.*)$", corpo)
        cond_txt, concl = m.group(1), m.group(2).strip()
        if cond_txt == "VERDADEIRO":
            conds, conector = [], "E"
        else:
            if " E " in cond_txt and " OU " in cond_txt:
                raise ValueError(f"{rid}: mistura E e OU na mesma regra")
            conector = "OU" if " OU " in cond_txt else "E"
            partes = re.split(r"\s+(?:E|OU)\s+", cond_txt)
            conds = []
            for p in partes:
                a, op, v = COND_RE.match(p).groups()
                conds.append((a, op, _val(v)))
        try:
            concl = float(concl.split("=")[-1])
        except ValueError:
            pass
        regras.append(Rule(rid, tipo, int(prio.replace("prio=", "")), conds, conector, concl))
    return sorted(regras, key=lambda r: r.prioridade)


class ForwardChainingEngine:
    """Motor de inferência. Não conhece o domínio: só aplica a base."""

    def __init__(self, rules_path, classes=None):
        self.rules_path = rules_path
        self.rules = load_rules(rules_path)
        self.classes = classes
        self.stats = {}   # distribuição de classes por regra (treino)

    def infer_one(self, fatos):
        for r in self.rules:            # agenda ordenada por prioridade
            if r.aplica(fatos):
                return r.conclusao, r.id
        raise RuntimeError("nenhuma regra disparou")

    def _iter(self, X):
        cols = list(X.columns)
        for row in X.itertuples(index=False):
            yield dict(zip(cols, row))

    def predict(self, X):
        out = [self.infer_one(f)[0] for f in self._iter(X)]
        return np.array(out)

    def fired(self, X):
        return [self.infer_one(f)[1] for f in self._iter(X)]

    # --- pontuação "tipo probabilidade" (só classificação) ---------------
    def fit_rule_stats(self, X, y, alpha=1.0):
        """Para cada regra, guarda a distribuição de classes das amostras de
        TREINO em que ela disparou (suavização de Laplace). Não é treinamento
        do modelo: é só a confiança empírica de cada regra."""
        fired = np.array(self.fired(X))
        for r in self.rules:
            sel = y.values[fired == r.id]
            cont = np.array([(sel == c).sum() for c in self.classes], float) + alpha
            self.stats[r.id] = (cont / cont.sum()).tolist()

    def predict_proba(self, X):
        return np.array([self.stats[i] for i in self.fired(X)])

    def save_stats(self, path):
        json.dump(self.stats, open(path, "w"))
