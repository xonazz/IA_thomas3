"""Gera reports/tables/resumo_final.csv e docs/relatorio.tex a partir das tabelas de reports/tables.
Uso: python -m src.make_report   (após python run_all.py)
Os números do texto são lidos das tabelas; trechos marcados com TODO são de autoria do aluno."""
import json
import os

import numpy as np
import pandas as pd

from .config import DATA, LOGS, ROOT, TABLES

T = lambda n: pd.read_csv(f"{TABLES}/{n}.csv")


def pt(x, nd=3):
    """Número no formato brasileiro (vírgula decimal)."""
    return f"{x:.{nd}f}".replace(".", ",")


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


REN = {"treino_tempo_medio_s": "treino_s", "inf_tempo_por_amostra_ms": "inf_ms/amostra", "inf_vazao_amostras_s": "vazao/s",
       "inf_pico_memoria_kb": "pico_inf_kb", "tamanho_disco_kb": "disco_kb", "treino_CO2_g": "CO2_treino_g"}


def tex_table(df, caption, label, fmt="{:.3f}"):
    df = df.rename(columns=REN)
    if "gap_relativo" in df:
        df = df.copy(); df.loc[df["gap_relativo"] > 100, "gap_relativo"] = np.nan   # RMSE de treino zero
    cols = list(df.columns)

    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if pd.isna(v) else esc(fmt.format(v).replace(".", ","))
        return esc(v)
    body = "\n".join(" & ".join(cell(v) for v in r) + r" \\" for r in df.itertuples(index=False))
    return (f"\\begin{{table}}[H]\\centering\\scriptsize\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{{'l' + 'r' * (len(cols) - 1)}}}\\toprule\n"
            f"{' & '.join(esc(c) for c in cols)} \\\\\\midrule\n{body}\n\\bottomrule\\end{{tabular}}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\\end{{table}}\n")


def resumo(p):
    m = T(f"metricas_{p}"); m = m[m.conjunto == "teste"].set_index("modelo")
    g = T(f"generalizacao_{p}").set_index("modelo"); e = T(f"eficiencia_{p}").set_index("modelo")
    cv = T(f"cv_final_{p}").set_index("modelo")
    df = m.drop(columns="conjunto").join(g[["gap", "diagnostico"]]).join(cv).join(
        e[["treino_tempo_medio_s", "inf_tempo_por_amostra_ms", "inf_vazao_amostras_s",
           "tamanho_disco_kb", "treino_CO2_g"]])
    df.insert(0, "problema", p)
    return df.reset_index()


def main():
    pd.concat([resumo("clf"), resumo("reg")]).to_csv(f"{TABLES}/resumo_final.csv", index=False)
    log = json.load(open(f"{TABLES}/log_clf.json")); amb = log["ambiente"]; logr = json.load(open(f"{TABLES}/log_reg.json"))
    mc = T("metricas_clf"); mr = T("metricas_reg")
    tc = mc[mc.conjunto == "teste"].set_index("modelo"); trc = mc[mc.conjunto == "treino"].set_index("modelo")
    tr_ = mr[mr.conjunto == "teste"].set_index("modelo")
    gc = T("generalizacao_clf").set_index("modelo"); gr = T("generalizacao_reg").set_index("modelo")
    cc = T("cv_final_clf").set_index("modelo"); cr = T("cv_final_reg").set_index("modelo")
    ec = T("eficiencia_clf").set_index("modelo"); er = T("eficiencia_reg").set_index("modelo")
    rfc = T("rf_curva_clf").set_index("n_arvores"); rfr = T("rf_curva_reg").set_index("n_arvores")
    ic = T("importancias_clf").set_index(T("importancias_clf").columns[0]); ir = T("importancias_reg").set_index(T("importancias_reg").columns[0])
    ctr = T("tree_criterio_clf").set_index("param_model__criterion").iloc[:, 0]
    crr = T("tree_criterio_reg").set_index("param_model__criterion").iloc[:, 0]
    ytr = pd.read_csv(f"{DATA}/clf_train.csv").Hypertension; yte = pd.read_csv(f"{DATA}/clf_test.csv").Hypertension
    ES, TR, RF, GB, BS = "Sist. Especialista", "Árvore (ajustada)", "Random Forest", "Gradient Boosting", "Baseline (constante)"
    U0, U1 = "Árvore sem poda (gini)", "Árvore sem poda (entropy)"
    R0, R1 = "Árvore sem poda (squared_error)", "Árvore sem poda (absolute_error)"
    tp, rp, gp = log["tree"]["best_params"], log["rf"]["best_params"], log["gb"]["best_params"]
    tpr, rpr, gpr = logr["tree"]["best_params"], logr["rf"]["best_params"], logr["gb"]["best_params"]

    def rules(name):
        t = open(f"{ROOT}/rules/{name}", encoding="utf-8").read()
        return "\n".join(l for l in t.splitlines() if not l.startswith("#"))

    ef_path = f"{TABLES}/esforco_humano.csv"
    if os.path.exists(ef_path):
        cron = tex_table(pd.read_csv(ef_path), "Esforço humano na elaboração das regras.", "tab:cron", "{:.2f}")
    else:
        cron = ("\\paragraph{Cronometragem.} \\textbf{TODO (aluno):} inserir a tabela de cronometragem (\\texttt{logs/cronometro\\_regras.csv}): "
                "no mínimo 3 sessões por problema, com data, horário, interrupções, nº de regras, tempo total, média, "
                "desvio-padrão e tempo médio por regra. Rode \\texttt{python -m src.cronometro} e depois \\texttt{python -m src.make\\_report}.")

    parts = []
    mt_cols = ["modelo", "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "F1_weighted",
               "Precision_pos", "Recall_pos", "F1_pos", "ROC_AUC", "PR_AUC"]
    parts.append(tex_table(mc[mc.conjunto == "teste"][mt_cols], "Classificação (Hypertension): métricas no conjunto de teste.", "tab:met_clf"))
    parts.append(tex_table(T("generalizacao_clf"), "Classificação: gap de generalização (F1 macro treino $-$ teste).", "tab:gap_clf"))
    parts.append(tex_table(T("cv_final_clf"), "Classificação: validação cruzada (5 folds, treino).", "tab:cv_clf", "{:.4f}"))
    parts.append(tex_table(T("eficiencia_clf")[["modelo", "treino_tempo_medio_s", "inf_tempo_por_amostra_ms", "inf_vazao_amostras_s",
                                                 "inf_pico_memoria_kb", "tamanho_disco_kb", "treino_CO2_g"]],
                           "Classificação: eficiência computacional (inferência no teste, 1000 amostras).", "tab:ef_clf", "{:.4g}"))
    parts.append(tex_table(mr[mr.conjunto == "teste"].drop(columns="conjunto"), "Regressão (Blood\\_Pressure\\_Systolic): métricas no teste.", "tab:met_reg", "{:.3f}"))
    parts.append(tex_table(T("generalizacao_reg"), "Regressão: gap de generalização (RMSE teste $-$ treino).", "tab:gap_reg", "{:.4g}"))
    parts.append(tex_table(T("cv_final_reg"), "Regressão: validação cruzada (5 folds, treino).", "tab:cv_reg", "{:.4f}"))
    parts.append(tex_table(T("eficiencia_reg")[["modelo", "treino_tempo_medio_s", "inf_tempo_por_amostra_ms", "inf_vazao_amostras_s",
                                                 "inf_pico_memoria_kb", "tamanho_disco_kb", "treino_CO2_g"]],
                           "Regressão: eficiência computacional.", "tab:ef_reg", "{:.4g}"))

    npos_tr, npos_te = int((ytr == "Yes").sum()), int((yte == "Yes").sum())
    ambs = f"{str(amb['so']).replace('_', chr(92) + '_')}, Python {amb['python']}, {amb['nucleos']} núcleo(s), {amb['ram_gb']} GB RAM"

    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[brazilian]{babel}
\usepackage{graphicx,booktabs,float,hyperref,geometry}\geometry{margin=2.2cm}
\graphicspath{{../reports/figures/}}
\begin{document}
\begin{titlepage}\centering
{\large UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE\\DCT1401 -- Inteligência Artificial\\}\vspace{2cm}
{\LARGE\bfseries Avaliação Comparativa de Modelos:\\Sistema Especialista, Árvore de Decisão e Ensemble Learning\par}
\vspace{2cm}Aluno: Jonas Jeronimo Cirilo Silva \quad Matrícula: 20240026090\\[1ex]
Dataset: \url{https://www.kaggle.com/datasets/mobeenfatimah/human-digital-twin-dataset-for-ai-and-healthcare}\\[1ex]
Repositório: \url{https://github.com/xonazz/IA\_thomas2}
\end{titlepage}

\section{Introdução}
Este relatório compara três famílias de modelos --- sistema especialista (regras SE-ENTÃO), árvore de decisão e ensembles
(Random Forest e Gradient Boosting) --- em dois problemas sobre o conjunto \emph{Human Digital Twin} (Kaggle), usando o arquivo
\texttt{digital\_twin\_profile.csv}: 5000 indivíduos sintéticos e 30 colunas (demografia, hábitos, sinais vitais e diagnósticos).
O conjunto é sintético, o que explica parte dos resultados discutidos adiante. Há 150 valores ausentes em cada uma de cinco colunas
(\texttt{BMI}, \texttt{Heart\_Rate}, \texttt{Physical\_Activity}, \texttt{Daily\_Water\_Intake\_L}, \texttt{Sleep\_Hours}), tratados por imputação dentro do \texttt{Pipeline}.
\textbf{Classificação:} prever \texttt{Hypertension} (Yes/No; 17,2\% de \emph{Yes}), com 24 features e \emph{sem} as colunas de pressão arterial
e sem diagnósticos, pois a pressão define a hipertensão e tornaria o problema trivial.
\textbf{Regressão:} prever \texttt{Blood\_Pressure\_Systolic} (mmHg), com 26 features: as mesmas, mais os diagnósticos
\texttt{Hypertension} e \texttt{Diabetes} (variáveis registradas à parte) e sem a pressão diastólica. Sem os diagnósticos, a regressão não tem
sinal algum (R$^2$ de validação cruzada negativo em teste preliminar). \textbf{TODO (aluno):} se os targets da atividade anterior forem outros, justificar aqui a mudança.

\section{Protocolo Experimental}
Divisão 80/20 com \texttt{random\_state=42} (estratificada na classificação), a mesma para os três tipos de modelo: 4000 amostras de treino e 1000 de teste
(classificação: """ + f"{npos_tr} e {npos_te}" + r""" casos \emph{Yes} no treino e no teste). O teste só foi usado na avaliação final; os
hiperparâmetros foram ajustados por validação cruzada de 5 folds no treino (\texttt{GridSearchCV} para a árvore e
\texttt{RandomizedSearchCV} para os ensembles; métrica de seleção: F1 macro e RMSE). Imputação e codificação ficam em um
\texttt{Pipeline} ajustado só no treino. Um \emph{baseline} constante (classe majoritária / média) serve de referência.
Ambiente de execução dos experimentos reportados: """ + ambs + r""".
\textbf{TODO (aluno):} reexecutar \texttt{python run\_all.py} na sua máquina e substituir pelo seu hardware. Energia estimada como potência
média assumida (45 W) $\times$ tempo, convertida com 45,8 gCO$_2$/kWh (fator do Brasil, Aula 08); o \texttt{codecarbon} não foi utilizado.

\section{Sistema Especialista}
A base de conhecimento fica em arquivo texto, separada do motor de inferência (encadeamento para frente, com resolução de
conflito por prioridade e regra padrão final). Cada regra usa só conjunções (E). Os limiares foram definidos observando apenas o treino
(taxas de hipertensão por faixa de IMC e idade) e conhecimento de domínio. Base de classificação (6 regras; tipos: relação, heurística, diretriz e estratégia):
\begin{verbatim}
""" + rules("regras_classificacao.txt") + r"""
\end{verbatim}
Base de regressão (5 regras; o consequente é a média da pressão sistólica no treino de cada combinação de diagnósticos):
\begin{verbatim}
""" + rules("regras_regressao.txt") + r"""
\end{verbatim}
O sistema especialista não produz probabilidades; para as curvas ROC e PR usou-se a distribuição de classes das amostras de treino em que
cada regra disparou (suavização de Laplace). É uma pontuação em degraus (seis valores distintos), o que limita as curvas.
""" + cron + r"""

\section{Árvore de Decisão}
Comparou-se Gini e entropia (classificação) e erro quadrático e erro absoluto (regressão), junto com \texttt{max\_depth},
\texttt{min\_samples\_leaf}, \texttt{ccp\_alpha} e (na classificação) \texttt{class\_weight}. Na classificação os dois critérios empataram na validação cruzada
(F1 macro """ + f"{pt(ctr['entropy'], 4)} e {pt(ctr['gini'], 4)}" + r"""); na regressão o erro quadrático foi um pouco melhor (RMSE """ + f"{pt(-crr['squared_error'], 3)} contra {pt(-crr['absolute_error'], 3)}" + r""").
Melhor árvore de classificação: entropia, \texttt{max\_depth=""" + f"{tp['max_depth']}, min\\_samples\\_leaf={tp['min_samples_leaf']}, ccp\\_alpha={str(tp['ccp_alpha']).replace('.', ',')}" + r"""}
(profundidade efetiva """ + f"{int(ec.loc[TR, 'profundidade'])}, {int(ec.loc[TR, 'nos_totais'])} nós, {int(ec.loc[TR, 'folhas'])} folhas" + r"""). Melhor árvore de regressão: erro quadrático,
profundidade """ + f"{int(er.loc[TR, 'profundidade'])}, {int(er.loc[TR, 'nos_totais'])} nós, {int(er.loc[TR, 'folhas'])} folhas" + r""".
As árvores sem restrição acertam 100\% do treino, mas o F1 macro no teste cai para """ + f"{pt(tc.loc[U0, 'F1_macro'], 2)} (gini) e {pt(tc.loc[U1, 'F1_macro'], 2)} (entropia)" + r""", pior que a árvore ajustada (""" + pt(tc.loc[TR, 'F1_macro'], 2) + r"""), e na regressão o R$^2$ no teste é negativo (""" + f"{pt(tr_.loc[R0, 'R2'], 2)}" + r"""): overfitting evidente.
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{arvore_clf.png}\caption{Árvore ajustada (classificação).}\end{figure}
\paragraph{Três regras extraídas da árvore de classificação e comparação com o especialista.} A árvore tem exatamente três folhas:
\begin{enumerate}
\item IMC $\le$ 28,05 $\Rightarrow$ No. Corresponde à regra R5 do especialista (IMC $<$ 25 $\Rightarrow$ No) e à regra padrão R6, com um limiar de IMC mais alto aprendido dos dados.
\item IMC $>$ 28,05 e idade $\le$ 50,5 $\Rightarrow$ No. O especialista não tem regra equivalente: suas regras R2 e R4 já preveem \emph{Yes} a partir de 45 anos, ou seja, a árvore é mais conservadora para quem tem 45 a 50 anos.
\item IMC $>$ 28,05 e idade $>$ 50,5 $\Rightarrow$ Yes. Resume as regras R1 a R3 do especialista (IMC $\ge$ 27,5 ou 30 com idade $\ge$ 45 ou 60) em uma só.
\end{enumerate}
O padrão é o mesmo: IMC e idade carregam o sinal. A árvore aprende limiares únicos e mais simples; o especialista tem mais regras, mas depende de quem as escreve.
Na regressão, a árvore usa \texttt{Hypertension} na raiz; os demais cortes (tempo de tela, altura, frequência cardíaca) são ruído.

\section{Ensemble Learning}
Random Forest e Gradient Boosting, ajustados por busca aleatória (\texttt{n\_estimators}, \texttt{max\_depth}, \texttt{max\_features},
\texttt{min\_samples\_leaf}, \texttt{learning\_rate}, \texttt{subsample}). Classificação: RF com """ + f"{rp['n_estimators']} árvores, profundidade {rp['max_depth']}, max\\_features={str(rp['max_features']).replace('.', ',')}, class\\_weight={rp['class_weight']}" + r""";
GB com """ + f"{gp['n_estimators']} estimadores, profundidade {gp['max_depth']}, taxa {str(gp['learning_rate']).replace('.', ',')}, subsample={str(gp['subsample']).replace('.', ',')}" + r""".
Regressão: RF com """ + f"{rpr['n_estimators']} árvores e profundidade {rpr['max_depth']}; GB com {gpr['n_estimators']} estimadores, profundidade {gpr['max_depth']} e taxa {str(gpr['learning_rate']).replace('.', ',')}" + r""".
\paragraph{Erro $\times$ número de árvores (Random Forest).} Na classificação, o erro de validação cruzada (1 $-$ F1 macro) cai de """ + f"{pt(rfc.loc[1, 'erro_cv'])} com 1 árvore para {pt(rfc.loc[10, 'erro_cv'])} com 10, {pt(rfc.loc[50, 'erro_cv'])} com 50 e {pt(rfc.loc[200, 'erro_cv'])} com 200" + r""", enquanto o erro de treino fica em torno de """ + pt(rfc.loc[200, 'erro_treino'], 2) + r""".
Na regressão o RMSE de validação vai de """ + f"{pt(rfr.loc[1, 'erro_cv'], 2)} para {pt(rfr.loc[50, 'erro_cv'], 2)} (50 árvores) e {pt(rfr.loc[200, 'erro_cv'], 2)} (200 árvores)" + r""".
A média de muitas árvores reduz a variância da árvore isolada; depois de $\sim$50 árvores o ganho é desprezível. A distância entre erro de treino e de validação, que não fecha, mostra que a floresta ainda memoriza parte do ruído (variância residual), e que o erro restante é sobretudo ruído irredutível.
\begin{figure}[H]\centering\includegraphics[width=.48\textwidth]{rf_curva_clf.png}
\includegraphics[width=.48\textwidth]{rf_curva_reg.png}\caption{Erro $\times$ número de árvores (RF).}\end{figure}
\paragraph{Importância das features.} Na classificação, a árvore usa apenas idade (""" + pt(ic.loc['Age', TR], 2) + r""") e IMC (""" + pt(ic.loc['BMI', TR], 2) + r"""); na RF e no GB, IMC e idade também lideram
(RF: """ + f"{pt(ic.loc['BMI', RF], 2)} e {pt(ic.loc['Age', RF], 2)}" + r"""), com o resto da importância diluída em variáveis sem relação real (peso, altura, renda, tempo de tela). São as variáveis das regras R1 a R5 do especialista.
Na regressão, \texttt{Hypertension} concentra a importância (árvore: """ + pt(ir.loc['Hypertension_No', TR], 2) + r"""; GB: """ + pt(ir.loc['Hypertension_No', GB] + ir.loc['Hypertension_Yes', GB], 2) + r"""; RF: """ + pt(ir.loc['Hypertension_No', RF] + ir.loc['Hypertension_Yes', RF], 2) + r"""), e o especialista usa exatamente essa variável (mais \texttt{Diabetes}, que quase não pesa).
\begin{figure}[H]\centering\includegraphics[width=.48\textwidth]{importancias_clf.png}
\includegraphics[width=.48\textwidth]{importancias_reg.png}\caption{Importância das features (impureza).}\end{figure}

\section{Resultados}
""" + "\n".join(parts) + r"""
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{matrizes_confusao.png}\caption{Matrizes de confusão (teste, classificação).}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{roc_pr.png}\caption{Curvas ROC e Precision-Recall da classe \emph{Yes} (teste).}\end{figure}
\begin{figure}[H]\centering\includegraphics[width=\textwidth]{real_vs_previsto_residuos.png}\caption{Real vs.\ previsto e resíduos (regressão).}\end{figure}
\paragraph{Escolha das métricas.} Na classificação, a Accuracy engana: o baseline que sempre responde \emph{No} já tem Accuracy de """ + pt(tc.loc[BS, 'Accuracy'], 3) + r""", maior que a do especialista (""" + pt(tc.loc[ES, 'Accuracy'], 3) + r""") e praticamente igual à da árvore ajustada, da RF e do GB, apesar de não acertar nenhum \emph{Yes}. Usou-se F1 macro como
métrica principal (baseline: """ + pt(tc.loc[BS, 'F1_macro'], 3) + r"""), por dar o mesmo peso à classe rara; ROC-AUC e PR-AUC complementam (a PR-AUC do baseline é a prevalência, 0,172). Em triagem clínica, o falso negativo (hipertenso não detectado) é o erro mais custoso, por isso o recall da classe positiva também é reportado.
Na regressão, a métrica principal é o RMSE, que penaliza erros grandes; MAE e MedAE (robusta a outliers) completam a análise. O baseline tem R$^2\approx0$ e RMSE """ + pt(tr_.loc[BS, 'RMSE'], 2) + r""" mmHg, e os melhores modelos chegam a R$^2\approx$ """ + pt(tr_.loc[GB, 'R2'], 2) + r""": as features explicam cerca de 20\% da variância, quase toda ela pelo diagnóstico de hipertensão; os resíduos têm dispersão uniforme (desvio de $\sim$13 mmHg dentro de cada grupo).

\section{Análise Comparativa e Discussão}
\subsection*{Classificação (Hypertension)}
\textbf{(a) Melhor desempenho.} F1 macro no teste: árvore ajustada e RF """ + pt(tc.loc[TR, 'F1_macro']) + r""", especialista """ + pt(tc.loc[ES, 'F1_macro']) + r""", GB """ + pt(tc.loc[GB, 'F1_macro']) + r""" e baseline """ + pt(tc.loc[BS, 'F1_macro']) + r""".
Na validação cruzada: árvore """ + f"{pt(cc.loc[TR, 'F1_macro_cv_media'])}$\\pm${pt(cc.loc[TR, 'F1_macro_cv_desvio'])}, RF {pt(cc.loc[RF, 'F1_macro_cv_media'])}$\\pm${pt(cc.loc[RF, 'F1_macro_cv_desvio'])}, GB {pt(cc.loc[GB, 'F1_macro_cv_media'])}$\\pm${pt(cc.loc[GB, 'F1_macro_cv_desvio'])}" + r""". Árvore e RF empatam; o especialista fica cerca de 0,02 abaixo, diferença que um teste com apenas """ + str(npos_te) + r""" casos \emph{Yes} não permite afirmar; o GB é pior no F1 (recall da classe positiva de """ + pt(tc.loc[GB, 'Recall_pos'], 2) + r""", pois com limiar 0,5 e sem pesos tende à classe majoritária), embora tenha a maior PR-AUC (""" + pt(tc.loc[GB, 'PR_AUC']) + r""") e ROC-AUC (""" + pt(tc.loc[GB, 'ROC_AUC']) + r"""). Todos os modelos ficam longe de uma boa discriminação (AUC $\le$ 0,69), coerente com o conjunto sintético de sinal fraco.
\textbf{(b) Generalização.} Gap de F1 macro: especialista """ + pt(gc.loc[ES, 'gap']) + r""", árvore ajustada """ + pt(gc.loc[TR, 'gap']) + r""" (bem ajustados), RF """ + pt(gc.loc[RF, 'gap']) + r""" e GB """ + pt(gc.loc[GB, 'gap']) + r""" (overfitting), árvores sem poda """ + f"{pt(gc.loc[U0, 'gap'], 2)} e {pt(gc.loc[U1, 'gap'], 2)}" + r""".
\textbf{(c) Eficiência.} O especialista e a árvore são os mais leves (""" + f"{pt(ec.loc[ES, 'tamanho_disco_kb'], 1)} KB e {pt(ec.loc[TR, 'tamanho_disco_kb'], 1)} KB" + r"""); a RF é a mais pesada (""" + f"{pt(ec.loc[RF, 'tamanho_disco_kb'] / 1024, 1)} MB, {int(ec.loc[RF, 'nos_totais'])} nós, {pt(ec.loc[RF, 'treino_tempo_medio_s'], 2)} s de treino" + r""") e o GB leva """ + pt(ec.loc[GB, 'treino_tempo_medio_s'], 2) + r""" s, contra """ + pt(ec.loc[TR, 'treino_tempo_medio_s'], 2) + r""" s da árvore.
A inferência do especialista (""" + pt(ec.loc[ES, 'inf_tempo_por_amostra_ms'], 3) + r""" ms/amostra) é um pouco mais lenta que a da árvore (""" + pt(ec.loc[TR, 'inf_tempo_por_amostra_ms'], 3) + r""") porque o motor é um laço em Python puro, sem vetorização. Energia e emissões são desprezíveis (ordem de $10^{-3}$ g CO$_2$ no treino dos ensembles). Há trade-off? Aqui não: os ensembles custam mais e não ganham em F1.
\textbf{(d) Custo do especialista.} O treino dos modelos de AM leva de """ + pt(ec.loc[TR, 'treino_tempo_medio_s'], 2) + r""" a """ + pt(ec.loc[RF, 'treino_tempo_medio_s'], 1) + r""" s; o especialista exige horas humanas de elaboração. \textbf{TODO (aluno):} inserir aqui o tempo médio por sessão e por regra obtido na cronometragem.
\textbf{(e) Interpretabilidade.} A árvore (3 folhas) e o especialista (6 regras) são totalmente auditáveis; o GB (""" + str(int(ec.loc[GB, 'nos_totais'])) + r""" nós) e a RF (""" + str(int(ec.loc[RF, 'nos_totais'])) + r""" nós) não. Em saúde, a explicabilidade importa, mas as relações são associativas, não causais.
\textbf{(f) Recomendação.} A árvore ajustada: F1 macro igual ao melhor, melhor generalização, 3 regras legíveis e custo mínimo. O especialista é uma alternativa equivalente quando o conhecimento clínico já está formalizado.

\subsection*{Regressão (Blood\_Pressure\_Systolic)}
\textbf{(a) Melhor desempenho.} RMSE no teste: GB """ + pt(tr_.loc[GB, 'RMSE'], 2) + r""", especialista """ + pt(tr_.loc[ES, 'RMSE'], 2) + r""", árvore """ + pt(tr_.loc[TR, 'RMSE'], 2) + r""", RF """ + pt(tr_.loc[RF, 'RMSE'], 2) + r""" e baseline """ + pt(tr_.loc[BS, 'RMSE'], 2) + r""".
Na validação cruzada: GB """ + f"{pt(cr.loc[GB, 'RMSE_cv_media'], 2)}$\\pm${pt(cr.loc[GB, 'RMSE_cv_desvio'], 2)}, árvore {pt(cr.loc[TR, 'RMSE_cv_media'], 2)}$\\pm${pt(cr.loc[TR, 'RMSE_cv_desvio'], 2)} e RF {pt(cr.loc[RF, 'RMSE_cv_media'], 2)}$\\pm${pt(cr.loc[RF, 'RMSE_cv_desvio'], 2)}" + r""". As diferenças (menos de 0,05 mmHg) são ínfimas diante do desvio (0,27): \emph{os quatro modelos empatam}, e todos superam o baseline por cerca de 1,7 mmHg.
\textbf{(b) Generalização.} Gap de RMSE: especialista """ + pt(gr.loc[ES, 'gap'], 3) + r""", árvore """ + pt(gr.loc[TR, 'gap'], 3) + r""", GB """ + pt(gr.loc[GB, 'gap'], 3) + r""" (bem ajustados), RF """ + pt(gr.loc[RF, 'gap'], 2) + r""" (overfitting leve, """ + pt(100 * gr.loc[RF, 'gap_relativo'], 1) + r"""\%) e árvores sem poda $\approx$ 19,7 (RMSE de treino zero).
\textbf{(c) Eficiência.} Mesma ordem da classificação: especialista (""" + pt(er.loc[ES, 'tamanho_disco_kb'], 1) + r""" KB) e árvore (""" + pt(er.loc[TR, 'tamanho_disco_kb'], 1) + r""" KB) contra RF (""" + pt(er.loc[RF, 'tamanho_disco_kb'] / 1024, 1) + r""" MB, """ + pt(er.loc[RF, 'treino_tempo_medio_s'], 2) + r""" s) e GB (""" + pt(er.loc[GB, 'tamanho_disco_kb'], 0) + r""" KB, """ + pt(er.loc[GB, 'treino_tempo_medio_s'], 2) + r""" s). Sem ganho de desempenho que justifique o custo.
\textbf{(d)} Ver o item (d) da classificação (preencher com o tempo cronometrado). \textbf{(e)} Mesma ordem de interpretabilidade; as 5 regras do especialista dizem o essencial (pressão média por diagnóstico). 
\textbf{(f) Recomendação.} O sistema especialista de 5 regras (ou, equivalentemente, uma árvore de profundidade 1 em \texttt{Hypertension}): mesmo RMSE dos ensembles, melhor generalização e custo mínimo.

\paragraph{Viés e variância.} Árvore sem poda: variância alta (treino perfeito, teste pior que o baseline). Especialista: viés alto por construção (poucas regras), mas com variância praticamente nula; aqui isso basta porque o sinal é simples. Árvore ajustada: compromisso intermediário.
RF: reduz variância por média (curva de erro), mas ainda apresenta gap; GB: árvores rasas e correção sequencial reduzem viés, porém na classificação o gap também foi alto.
\paragraph{Vantagens e desvantagens.} Especialista: transparente e barato de executar, mas sem aprendizado, manutenção manual e dependente do engenheiro de conhecimento. Árvore: interpretável, porém instável e propensa a overfitting se não for podada. Ensembles: mais robustos em geral, porém menos interpretáveis, maiores e, neste conjunto de sinal fraco, sem benefício.
\paragraph{Implantação.} Em dispositivo de borda, o especialista (menos de 1 KB e $\sim$0,3 MB de pico na inferência) ou a árvore (9 a 10 KB); em servidor, GB ou RF só se houver ganho comprovado, o que aqui não ocorreu.

\section{Considerações Finais}
\textbf{Limitações:} conjunto sintético com sinal fraco (AUC $\le$ 0,69; R$^2\approx$ 0,2), o que reduz a diferença entre modelos; uma única divisão treino/teste e apenas """ + str(npos_te) + r""" casos \emph{Yes} no teste; na regressão, \texttt{Hypertension} e \texttt{Diabetes} são diagnósticos usados como features, e a pressão sistólica quase se reduz a uma função do primeiro;
regras escritas após inspecionar o treino; a validação cruzada do especialista não foi feita (opcional); o GB não teve ajuste de limiar nem pesos de classe; tempos medidos em ambiente de 2 núcleos, energia estimada por potência assumida e memória medida com \texttt{tracemalloc}, que não captura toda alocação em C.
Só o arquivo \texttt{digital\_twin\_profile.csv} foi usado; os demais arquivos do conjunto não foram combinados.
\textbf{Aprendizados:} em dados de sinal fraco, modelos simples empatam com ensembles, e comparar com um baseline constante é essencial (a Accuracy sozinha engana).
\textbf{TODO (aluno):} acrescentar dificuldades pessoais encontradas.

\section{Repositório}
Repositório: \url{https://github.com/xonazz/IA\_thomas2}. Estrutura: \texttt{src/} (código), \texttt{rules/} (bases de regras), \texttt{notebooks/} (03 a 08), \texttt{logs/} (cronometragem),
\texttt{reports/tables} e \texttt{reports/figures} (resultados) e \texttt{docs/} (este relatório).
\end{document}
"""
    open(f"{ROOT}/docs/relatorio.tex", "w", encoding="utf-8").write(tex)
    print("ok")


if __name__ == "__main__":
    main()
