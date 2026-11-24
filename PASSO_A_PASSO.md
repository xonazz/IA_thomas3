# Passo a passo (execute dentro da pasta projeto/)

1. unzip projeto.zip && cd projeto
2. pip install -r requirements.txt        (se o codecarbon falhar, remova-o do arquivo)
3. python -m src.data_prep                (datasets e splits; dataset: digital_twin_profile.csv)
4. Elabore/revise as regras e cronometre (mín. 3 sessões por problema):
   - edite rules/regras_classificacao.txt e rules/regras_regressao.txt
   - python -m src.sessao clf   |   python -m src.sessao reg
5. Ajuste POTENCIA_ASSUMIDA_W em src/config.py
6. python run_all.py                      (~10 min; rode UMA execução por vez, senão os tempos são distorcidos)
7. python -m src.cronometro               (média, desvio, tempo por regra)
8. Edite os TODO em src/make_report.py (item (d), dificuldades, hardware) e rode:
   python -m src.make_report              (gera docs/relatorio.tex e resumo_final.csv)
9. Confira no texto do relatório os tempos/tamanhos (dependem do hardware)
10. Compile docs/relatorio.tex -> salve como docs/relatorio_final.pdf
11. Abra notebooks/03..08 no Jupyter e confira se rodam
12. Suba tudo para o repositório público no GitHub (https://github.com/xonazz/IA_thomas2)
