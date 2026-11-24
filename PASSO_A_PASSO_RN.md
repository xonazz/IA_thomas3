# Passo a passo: atividade de Redes Neurais (execute dentro da pasta projeto/)

## Opção A: Google Colab (mais fácil; já tem PyTorch)
1. Suba a pasta `projeto` para o seu GitHub (ou envie o zip para o Colab) e, numa célula:
   `!git clone https://github.com/xonazz/IA_thomas2 && %cd IA_thomas2/projeto`
2. `!pip install psutil` (os outros pacotes o Colab já tem)
3. `!python -m src.data_prep`
4. Teste de fumaça: `!RN_RAPIDO=1 python run_rn.py`  (~2 min; não mistura com os resultados reais)
5. Rodada completa: `!python run_rn.py`  (demora; se o Colab cair, rode de novo que ele continua de onde parou)
6. Baixe as pastas `logs/`, `models/`, `reports/`, `data/novas_amostras_*.csv` e `docs/relatorio_rn.tex`
   (ex.: `!zip -r resultados.zip logs models reports docs data/novas_amostras_*.csv`) e coloque no repositório.

## Opção B: seu computador
1. `pip install -r requirements.txt` (se o codecarbon falhar, remova-o do arquivo)
2. Mesmos passos 3 a 5 acima, sem o `!`. Feche outros programas durante a rodada (os tempos medidos dependem disso).

## Depois de rodar
7. Processador e potência já estão configurados em `src/rn_config.py`: no seu PC, AMD Ryzen 5 7430U com 15 W
   (TDP); no Colab, a potência padrão de `src/config.py` (45 W). Nada a fazer aqui.
8. Abra `docs/relatorio_rn.tex` e:
   - procure `% REVISAR` e confira se o texto automático bate com os gráficos;
   - complete o `TODO` das considerações finais;
   - confira a recomendação final (item f).
9. Compile no Overleaf (envie também `reports/figures/rn/`) e salve como `docs/relatorio_final.pdf`.
10. Preencha "Principais resultados" no README e rode os notebooks `notebooks/rn/03..08`.
11. Faça o commit de tudo (incluindo `models/*.pt` e `logs/hiperparametros.csv`).

Para usar 5 sementes em vez de 3: `RN_SEEDS=5 python run_rn.py` (demora ~5/3 do tempo).
