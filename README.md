# Avaliação Comparativa de Modelos: Sistema Especialista, Árvore de Decisão e Ensemble

**Aluno:** Jonas Jeronimo Cirilo Silva | **Matrícula:** 20240026090
**Dataset:** [Human Digital Twin Dataset for AI & Healthcare (Kaggle)](https://www.kaggle.com/datasets/mobeenfatimah/human-digital-twin-dataset-for-ai-and-healthcare?select=digital_twin_profile.csv) — arquivo `data/raw/digital_twin_profile.csv`
**Repositório:** https://github.com/xonazz/IA_thomas2

## Descrição
Compara sistema especialista (regras SE-ENTÃO), árvore de decisão e ensembles (Random Forest e Gradient Boosting) com as
métricas da Aula 08 (desempenho, generalização e eficiência computacional).
- Classificação: `Hypertension` (Yes/No, 17,2% Yes), 24 features, sem colunas de pressão arterial nem diagnósticos.
- Regressão: `Blood_Pressure_Systolic`, 26 features (inclui os diagnósticos Hypertension e Diabetes; sem pressão diastólica).
- Divisão 80/20, `random_state=42` (estratificada na classificação); ajuste por CV de 5 folds só no treino.

## Como executar (a partir da raiz do projeto)
1. `pip install -r requirements.txt`
2. `python -m src.data_prep`   (datasets e splits)
3. Elaborar/revisar as regras em `rules/` e cronometrar: `python -m src.sessao clf` e `python -m src.sessao reg` (mín. 3 sessões cada)
4. `python run_all.py`          (modelos, métricas, figuras e eficiência -> `reports/`; ~10 min)
5. `python -m src.cronometro`   (estatísticas das sessões) e `python -m src.make_report` (gera `docs/relatorio.tex`)
6. Notebooks `03` a `08` (exibem os resultados gerados), nesta ordem.
Ajuste `POTENCIA_ASSUMIDA_W` em `src/config.py` e informe o hardware no relatório.

## Principais resultados (teste; ambiente de 2 núcleos, 7,8 GB)
- Classificação (F1 macro): árvore ajustada 0,682 | Random Forest 0,682 | especialista 0,660 | Gradient Boosting 0,606 | baseline 0,453.
  Árvore e RF empatam na validação cruzada (0,703 ± 0,011 e 0,696 ± 0,008). Todos têm AUC ≤ 0,69 (conjunto sintético, sinal fraco).
- Regressão (RMSE, mmHg): GB 13,74 | especialista 13,75 | árvore 13,75 | RF 13,80 | baseline 15,43 (R² ≈ 0,20). Empate técnico: quase todo o sinal vem do diagnóstico de hipertensão.
- Árvores sem poda: overfitting claro (treino perfeito, teste pior que o baseline na regressão).
- Tabela completa: `reports/tables/resumo_final.csv`. Relatório: `docs/relatorio.tex`.
- Esforço humano das regras: `logs/cronometro_regras.csv` (a preencher pelo aluno).

---

# Atividade 2: Redes Neurais com PyTorch (Classificação e Regressão)

MLP em PyTorch para os **mesmos targets, features e divisão treino/teste** da atividade acima, com estudo de
hiperparâmetros, persistência em `.pt`, inferência em ambiente limpo e as métricas da Aula 08.

## Como executar (a partir da pasta `projeto/`)
1. `pip install -r requirements.txt` (inclui `torch`)
2. `python -m src.data_prep` (gera `data/processed`, mesma divisão 80/20, `random_state=42`)
3. Teste rápido (~2 min, pastas separadas): `RN_RAPIDO=1 python run_rn.py`
4. Execução completa: `python run_rn.py` (estudo de hiperparâmetros + modelo final + relatório).
   Se for interrompido, rode de novo: o que já está em `logs/hiperparametros.csv` é pulado.
5. Compile `docs/relatorio_rn.tex` (Overleaf ou `pdflatex`) e salve como `docs/relatorio_final.pdf`
6. Notebooks em `notebooks/rn/` (03 a 08), nesta ordem: exibem os resultados gerados.

## Inferência com o modelo salvo
```
python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/novas_amostras_clf.csv
python src/inference.py --modelo models/modelo_regressao.pt --entrada data/novas_amostras_reg.csv --saida previsoes.csv
```
A entrada é um CSV com as colunas originais do dataset; o checkpoint guarda pesos, arquitetura, scaler,
codificação das categóricas, ordem das features, classes e limiar (carregado com `weights_only=True`).

## Organização (atividade 2)
| arquivo | conteúdo |
|---|---|
| `src/rn_models.py` | classe `MLP` parametrizável |
| `src/rn_train.py` | laço de treino (autograd, DataLoader, early stopping), sementes, métricas |
| `src/rn_prep.py` | pré-processamento ajustado no treino e exportável para o checkpoint |
| `src/rn_study.py` | estudo de um fator por vez, mapa de calor lr x batch, busca aleatória |
| `src/rn_final.py` | modelo final, limiar, `.pt`, fidelidade, K-fold, eficiência, TorchScript/quantização |
| `src/inference.py` | inferência independente a partir do `.pt` |
| `src/rn_report.py` | gera `docs/relatorio_rn.tex` com os números das tabelas |
| `logs/hiperparametros.csv` | todas as execuções (configuração x semente) |
| `reports/tables/rn`, `reports/figures/rn` | tabelas e gráficos |
| `models/` | `modelo_classificacao.pt`, `modelo_regressao.pt` (e os baselines) |

## Principais resultados
TODO: preencher após a execução com `reports/tables/rn/resumo_final.csv`
(métrica principal no teste, hiperparâmetros mais críticos e diferença máxima do modelo recarregado).
