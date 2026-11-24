"""Configuração da atividade de Redes Neurais (MLP em PyTorch).

Reaproveita caminhos, targets, features e a divisão treino/teste da atividade anterior (src/config.py).
"""
import os

from .config import FATOR_CO2_G_KWH, LOGS, MODELS, ROOT, SEED  # noqa: F401
from .config import POTENCIA_ASSUMIDA_W as _POT_PADRAO

# Hardware do aluno: AMD Ryzen 5 7430U (6 núcleos / 12 threads, TDP padrão de 15 W).
# No Google Colab a máquina é outra, então lá se mantém a potência padrão de src/config.py.
NO_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
CPU_NOME = "Google Colab (máquina virtual)" if NO_COLAB else "AMD Ryzen 5 7430U (6 núcleos / 12 threads, TDP 15 W)"
POTENCIA_ASSUMIDA_W = _POT_PADRAO if NO_COLAB else 15.0

# Sementes: cada configuração é treinada com todas elas (média ± desvio). Mínimo exigido: 3.
# Para usar 5 sementes:  RN_SEEDS=5 python run_rn.py
N_SEEDS = int(os.environ.get("RN_SEEDS", 3))
SEEDS = [SEED + i for i in range(N_SEEDS)]

VAL_SIZE = 0.20          # 20% do treino vira validação (estratificado na classificação)
N_REP_INFERENCIA = 30
K_FOLDS = 5

# Pastas próprias desta atividade (não sobrescrevem as da atividade anterior)
TABLES = os.path.join(ROOT, "reports", "tables", "rn")
FIGS = os.path.join(ROOT, "reports", "figures", "rn")
for d in (TABLES, FIGS, MODELS, LOGS):
    os.makedirs(d, exist_ok=True)
LOG_HP = os.path.join(LOGS, "hiperparametros.csv")

# Configuração de referência (baseline) -- igual à sugerida no enunciado
BASELINE = {
    "neurons": 32, "layers": 2, "activation": "ReLU", "optimizer": "Adam", "lr": 1e-3,
    "batch": 32, "epochs": 200, "patience": 20, "weight_decay": 0.0, "dropout": 0.0,
    "loss": "padrao", "scale": True,
}
# loss "padrao": MSE na regressão; BCE com pos_weight (n_neg/n_pos) na classificação.

# Estudo de um fator por vez: hiperparâmetro -> valores testados
FULL_BATCH = "total"
STUDY = {
    # obrigatórios
    "neurons": [2, 4, 8, 16, 32, 64, 128],
    "layers": [1, 2, 3, 5],
    "activation": ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU"],
    "optimizer": ["SGD", "SGD_momentum", "RMSprop", "Adam", "AdamW"],
    "lr": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "batch": [8, 32, 128, 512, FULL_BATCH],
    # opcionais (4 escolhidos)
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "loss": {"reg": ["MSE", "MAE", "Huber"], "clf": ["BCE", "BCE_pos_weight"]},
    "scale": [True, False],
}
OBRIGATORIOS = ["neurons", "layers", "activation", "optimizer", "lr", "batch"]
NOMES = {"neurons": "Neurônios por camada", "layers": "Camadas ocultas", "activation": "Função de ativação",
         "optimizer": "Otimizador", "lr": "Taxa de aprendizado", "batch": "Tamanho do batch",
         "weight_decay": "Weight decay (L2)", "dropout": "Dropout", "loss": "Função de perda",
         "scale": "Padronização das features"}

# Varredura 2D (interação): taxa de aprendizado x tamanho do batch
HEATMAP = {"lr": [1e-4, 1e-3, 1e-2, 1e-1], "batch": [8, 32, 128, 512]}

# Busca aleatória final (sobre a validação)
N_RANDOM = 20
RANDOM_SPACE = {
    "neurons": [16, 32, 64, 128], "layers": [1, 2, 3], "activation": ["ReLU", "LeakyReLU", "GELU", "Tanh"],
    "optimizer": ["Adam", "AdamW", "RMSprop"], "lr": (3e-4, 1e-2),  # log-uniforme
    "batch": [32, 64, 128], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3], "dropout": [0.0, 0.1, 0.3],
}

# Modo rápido (teste de fumaça, ~2 min): RN_RAPIDO=1 python run_rn.py
# Usa poucas épocas/valores e grava em pastas separadas (não mistura com os resultados reais).
RAPIDO = os.environ.get("RN_RAPIDO", "0") == "1"
if RAPIDO:
    SEEDS = SEEDS[:2]
    BASELINE.update(epochs=4, patience=2)
    STUDY = {k: ({p: v[:2] for p, v in vals.items()} if isinstance(vals, dict) else vals[:2])
             for k, vals in STUDY.items()}
    HEATMAP = {"lr": [1e-3, 1e-2], "batch": [32, 128]}
    N_RANDOM, K_FOLDS, N_REP_INFERENCIA = 2, 2, 3
    TABLES, FIGS = TABLES + "_rapido", FIGS + "_rapido"
    LOG_HP = os.path.join(LOGS, "hiperparametros_rapido.csv")
    for d in (TABLES, FIGS):
        os.makedirs(d, exist_ok=True)
SUFIXO_MODELO = "_rapido" if RAPIDO else ""

# Métrica principal (validação) de cada problema
PRIM = {"clf": "F1_macro", "reg": "RMSE"}
MAIOR_MELHOR = {"clf": True, "reg": False}
