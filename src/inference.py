"""Inferência com o modelo salvo (.pt), em ambiente limpo: não usa nada do treinamento, só o checkpoint.

Exemplos (a partir da raiz do projeto):
    python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/novas_amostras_clf.csv
    python src/inference.py --modelo models/modelo_regressao.pt --entrada data/novas_amostras_reg.csv --saida previsoes.csv

A entrada é um CSV com as colunas ORIGINAIS do dataset (texto nas categóricas, valores ausentes permitidos).
O script aplica o mesmo pré-processamento salvo no checkpoint (imputação, one-hot, ordinal e padronização com
média/desvio do treino), coloca a rede em modo de avaliação (eval) e devolve as previsões.
"""
import argparse
import os
import sys

import pandas as pd
import torch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)
from src.rn_models import build  # noqa: E402  (só a definição da classe MLP)
from src.rn_prep import Encoder  # noqa: E402  (só as regras de codificação; os valores vêm do checkpoint)


class Preditor:
    def __init__(self, caminho):
        # weights_only=True: o arquivo só pode conter tensores e tipos simples (não executa pickle arbitrário)
        # map_location="cpu": carrega na CPU mesmo que o treino tenha sido feito em GPU
        self.ckpt = torch.load(caminho, map_location="cpu", weights_only=True)
        ck = self.ckpt
        self.problema = ck["problema"]
        self.modelo = build(ck["arch"])                 # recria a arquitetura
        self.modelo.load_state_dict(ck["state_dict"])   # carrega pesos e vieses
        self.modelo.eval()                              # desativa dropout (modo de inferência)
        self.enc = Encoder.from_dict(ck["encoder"])     # medianas, modas e categorias do TREINO

    def tensor_entrada(self, df):
        faltando = [c for c in self.enc.raw_columns if c not in df.columns]
        if faltando:
            raise ValueError(f"colunas ausentes na entrada: {faltando}")
        x = torch.tensor(self.enc.encode(df[self.enc.raw_columns]), dtype=torch.float32)
        return (x - self.ckpt["x_media"]) / self.ckpt["x_desvio"]   # mesma padronização do treino

    def saida_bruta(self, df):
        with torch.inference_mode():
            return self.modelo(self.tensor_entrada(df)).squeeze(1)

    def prever(self, df):
        z = self.saida_bruta(df)
        ck = self.ckpt
        with torch.inference_mode():
            return self._pos(z, ck)

    def _pos(self, z, ck):
        if self.problema == "reg":
            y = z * ck["y_desvio"] + ck["y_media"]       # volta para mmHg
            return pd.DataFrame({f"{ck['target']}_previsto": y.numpy()})
        prob = torch.sigmoid(z)                         # probabilidade só na inferência
        classe = [ck["classes"][int(v)] for v in (prob >= ck["limiar"]).long()]
        return pd.DataFrame({f"prob_{ck['classes'][1]}": prob.numpy(), f"{ck['target']}_previsto": classe})


def main():
    ap = argparse.ArgumentParser(description="Inferência com o MLP salvo em .pt")
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--entrada", required=True, help="CSV com as colunas originais das features")
    ap.add_argument("--saida", default=None, help="CSV de saída (opcional)")
    a = ap.parse_args()
    pred = Preditor(a.modelo)
    df = pd.read_csv(a.entrada)
    out = pred.prever(df)
    ck = pred.ckpt
    print(f"modelo: {a.modelo} | problema: {pred.problema} | arquitetura: {ck['arch']}")
    if pred.problema == "clf":
        print(f"limiar de decisão (escolhido na validação): {ck['limiar']:.2f}")
    print(out.head(20).to_string())
    if a.saida:
        out.to_csv(a.saida, index=False)
        print(f"previsões salvas em {a.saida} ({len(out)} linhas)")


if __name__ == "__main__":
    main()
