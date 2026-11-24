"""MLP parametrizável em PyTorch (Seção 2 do enunciado)."""
import torch.nn as nn


class MLP(nn.Module):
    """Perceptron multicamadas: [Linear -> ativação (-> Dropout)] x n_camadas -> Linear (saída em logits).

    Regressão: 1 neurônio linear (sem ativação) + MSELoss.
    Classificação binária: 1 neurônio sem sigmoide (logit) + BCEWithLogitsLoss.
    """

    def __init__(self, in_dim, out_dim=1, hidden=(32, 32), activation="ReLU", dropout=0.0):
        super().__init__()
        act = getattr(nn, activation)                  # ex.: nn.ReLU, nn.Tanh, nn.GELU
        camadas, d = [], in_dim
        for h in hidden:
            camadas += [nn.Linear(d, h), act()]
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h
        camadas.append(nn.Linear(d, out_dim))           # saída em logits / valor contínuo
        self.net = nn.Sequential(*camadas)

    def forward(self, x):
        return self.net(x)


def arch(cfg, in_dim):
    """Hiperparâmetros da arquitetura (vão para o checkpoint para recriar o modelo)."""
    return {"in_dim": int(in_dim), "out_dim": 1, "hidden": [int(cfg["neurons"])] * int(cfg["layers"]),
            "activation": str(cfg["activation"]), "dropout": float(cfg["dropout"])}


def build(a):
    return MLP(a["in_dim"], a["out_dim"], tuple(a["hidden"]), a["activation"], a["dropout"])


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macs(a):
    """Multiplicações-acumulações por amostra (só as camadas lineares; ativações são desprezíveis)."""
    dims = [a["in_dim"]] + list(a["hidden"]) + [a["out_dim"]]
    return sum(i * o for i, o in zip(dims[:-1], dims[1:]))


def n_params_arch(a):
    dims = [a["in_dim"]] + list(a["hidden"]) + [a["out_dim"]]
    return sum(i * o + o for i, o in zip(dims[:-1], dims[1:]))
