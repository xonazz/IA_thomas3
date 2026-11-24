"""Gera docs/relatorio_rn.tex (relatório da atividade de Redes Neurais) e reports/tables/rn/resumo_final.csv.

Uso: python run_rn.py --etapa relatorio   (depois das etapas "estudo" e "final")
Todos os números do texto são lidos das tabelas em reports/tables/rn. Os trechos de evidência e conclusão
são montados a partir dos resultados; revise-os (comentários "% REVISAR" no .tex) e complete os TODO.
"""
import json
import os

import numpy as np
import pandas as pd

from . import rn_config as C
from .config import ROOT
from .rn_prep import Encoder, split_val

PROBS = ["clf", "reg"]
NP = {"clf": "classificação", "reg": "regressão"}
MET = {"clf": "F1 macro", "reg": "RMSE (mmHg)"}


# ---------------------------------------------------------------- utilitários
def T(n, **kw):
    return pd.read_csv(f"{C.TABLES}/{n}.csv", **kw)


def J(n):
    return json.load(open(f"{C.TABLES}/{n}.json"))


def pt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}".replace(".", ",")


def pm(m, s, nd=3):
    return f"{pt(m, nd)} $\\pm$ {pt(0.0 if not np.isfinite(s) else s, nd)}" if np.isfinite(m) else "divergiu"


def esc(s):
    return (str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$"))


def rot(v):
    try:
        return f"{float(v):g}".replace(".", ",")
    except (TypeError, ValueError):
        return esc({"True": "com", "False": "sem", "total": "total (lote inteiro)"}.get(str(v), str(v)))


def tabela(cab, linhas, caption, label, alin=None):
    alin = alin or ("l" + "r" * (len(cab) - 1))
    corpo = "\n".join(" & ".join(str(c) for c in l) + r" \\" for l in linhas)
    return (f"\\begin{{table}}[H]\\centering\\scriptsize\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{{alin}}}\\toprule\n{' & '.join(cab)} \\\\\\midrule\n{corpo}\n\\bottomrule\\end{{tabular}}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\\end{{table}}\n")


def figura(nome, caption, largura="0.95"):
    return (f"\\begin{{figure}}[H]\\centering\\includegraphics[width={largura}\\linewidth]{{{nome}}}"
            f"\\caption{{{caption}}}\\end{{figure}}\n")


def maior(p):
    return C.MAIOR_MELHOR[p]


# ---------------------------------------------------------------- estudo de hiperparâmetros
class Estudo:
    def __init__(self, p):
        self.p = p
        self.df = T(f"estudo_{p}", dtype={"valor": str})
        self.sens = T(f"sensibilidade_{p}", dtype={"melhor_valor": str, "pior_valor": str})

    def d(self, param):
        return self.df[(self.df.estudo == "ofat") & (self.df.param == param)].reset_index(drop=True)

    def get(self, param, valor):
        d = self.d(param)
        r = d[d.valor == str(valor)]
        return None if r.empty else r.iloc[0]

    def ext(self, param):
        """melhor e pior linha (valores finitos)"""
        d = self.d(param)
        f = d[np.isfinite(d.val_media)]
        if f.empty:
            return None, None, d
        b = f.loc[f.val_media.idxmax() if maior(self.p) else f.val_media.idxmin()]
        w = f.loc[f.val_media.idxmin() if maior(self.p) else f.val_media.idxmax()]
        return b, w, d

    def faixa(self, param):
        """valores estatisticamente empatados com o melhor (diferença <= soma dos desvios)"""
        b, _, d = self.ext(param)
        if b is None:
            return []
        ok = []
        for r in d.itertuples():
            if np.isfinite(r.val_media) and abs(r.val_media - b.val_media) <= np.nan_to_num(b.val_desvio) + np.nan_to_num(r.val_desvio):
                ok.append(rot(r.valor))
        return ok

    def tabela(self, param):
        d = self.d(param)
        cab = ["valor", "parâm.", f"{MET[self.p]} treino", f"{MET[self.p]} validação", "gap", "época parada", "tempo (s)", "diagnóstico"]
        linhas = [[rot(r.valor), int(r.n_params), pm(r.treino_media, r.treino_desvio), pm(r.val_media, r.val_desvio),
                   pt(r.gap), pt(r.epoca_parada, 1), pt(r.tempo_s, 1), r.diagnostico] for r in d.itertuples()]
        return tabela(cab, linhas, f"{C.NOMES[param]} -- {NP[self.p]} (média $\\pm$ desvio entre {len(C.SEEDS)} sementes; "
                      f"gap = treino $-$ validação{'' if self.p == 'clf' else ' em erro: validação $-$ treino'}).",
                      f"tab:{self.p}_{param}")

    def evidencia(self, param):
        b, w, d = self.ext(param)
        if b is None:
            return f"Na {NP[self.p]}, todas as configurações divergiram."
        amp = abs(b.val_media - w.val_media)
        ruido = np.nan_to_num(b.val_desvio) + np.nan_to_num(w.val_desvio)
        s = (f"Na {NP[self.p]}, o melhor {MET[self.p]} de validação foi {pm(b.val_media, b.val_desvio)} ({rot(b.valor)}) "
             f"e o pior {pm(w.val_media, w.val_desvio)} ({rot(w.valor)}); a diferença de {pt(amp)} é "
             f"{'maior' if amp > ruido else 'menor'} que a soma dos desvios ({pt(ruido)}), logo "
             f"{'o efeito é real' if amp > ruido else 'o efeito está dentro do ruído entre sementes e não deve ser tratado como real'}. ")
        div = d[~np.isfinite(d.val_media) | (d.divergiu > 0)]
        if len(div):
            s += f"Houve divergência (perda NaN/infinita) em: {', '.join(rot(v) for v in div.valor)}. "
        g = d[np.isfinite(d.gap)]
        if len(g):
            gi, ga = g.loc[g.gap.idxmin()], g.loc[g.gap.idxmax()]
            s += f"O gap variou de {pt(gi.gap)} ({rot(gi.valor)}) a {pt(ga.gap)} ({rot(ga.valor)}). "
        ei, ea = d.loc[d.epoca_parada.idxmin()], d.loc[d.epoca_parada.idxmax()]
        s += (f"A melhor época média foi de {pt(ei.epoca_parada, 0)} ({rot(ei.valor)}) a {pt(ea.epoca_parada, 0)} ({rot(ea.valor)}), "
              f"e o tempo médio de treino de {pt(d.tempo_s.min(), 1)} s a {pt(d.tempo_s.max(), 1)} s.")
        return s

    def conclusao(self, param):
        f = self.faixa(param)
        b, _, _ = self.ext(param)
        s = (f"Na {NP[self.p]}, os valores estatisticamente empatados com o melhor ({rot(b.valor) if b is not None else '--'}) foram: "
             f"{', '.join(f) if f else '--'}. {self.hipotese(param)}")
        return s

    # verificação automática da hipótese (sempre com os números medidos)
    def hipotese(self, param):
        p, d = self.p, self.d(param)
        mx = maior(p)
        pior = (lambda a, b: a < b) if mx else (lambda a, b: a > b)   # a é pior que b?
        b, _, _ = self.ext(param)
        try:
            if param == "neurons":
                lo, hi = d.iloc[0], d.iloc[-1]
                return (f"Com {rot(lo.valor)} neurônios o diagnóstico foi \\emph{{{lo.diagnostico}}} (validação {pt(lo.val_media)}), "
                        f"e com {rot(hi.valor)} o gap foi {pt(hi.gap)} contra {pt(lo.gap)} com {rot(lo.valor)}: a hipótese de que mais "
                        f"capacidade abre o gap {'se confirmou' if hi.gap > lo.gap else 'não se confirmou'} neste dataset.")
            if param == "layers":
                lo, hi = d.iloc[0], d.iloc[-1]
                return (f"Passar de {rot(lo.valor)} para {rot(hi.valor)} camada(s) multiplicou os parâmetros de {int(lo.n_params)} para "
                        f"{int(hi.n_params)} e {'piorou' if pior(hi.val_media, lo.val_media) else 'não piorou'} a validação "
                        f"({pt(lo.val_media)} $\\to$ {pt(hi.val_media)}).")
            if param == "activation":
                sg, rl = self.get(param, "Sigmoid"), self.get(param, "ReLU")
                if sg is not None and rl is not None:
                    return (f"A Sigmoid levou {pt(sg.epoca_parada, 0)} épocas até o melhor ponto, contra {pt(rl.epoca_parada, 0)} da ReLU, "
                            f"com validação {pt(sg.val_media)} contra {pt(rl.val_media)}; a hipótese de convergência mais lenta por saturação "
                            f"{'se confirmou' if sg.epoca_parada > rl.epoca_parada else 'não se confirmou'}.")
            if param == "optimizer":
                sgd, adam = self.get(param, "SGD"), self.get(param, "Adam")
                if sgd is not None and adam is not None:
                    return (f"O SGD simples ficou em {pt(sgd.val_media)} (melhor época {pt(sgd.epoca_parada, 0)}) contra {pt(adam.val_media)} "
                            f"do Adam (época {pt(adam.epoca_parada, 0)}); a hipótese de que o SGD sem momento converge mais devagar com "
                            f"$\\eta=10^{{-3}}$ {'se confirmou' if (sgd.epoca_parada > adam.epoca_parada or pior(sgd.val_media, adam.val_media)) else 'não se confirmou'}.")
            if param == "lr":
                lo, hi = d.iloc[0], d.iloc[-1]
                return (f"Com $\\eta$ = {rot(lo.valor)} o treino rodou em média {pt(lo.epocas_rodadas, 0)} de {C.BASELINE['epochs']} épocas "
                        f"({'atingiu o limite: convergência lenta' if lo.epocas_rodadas >= C.BASELINE['epochs'] - 1 else 'parou antes do limite'}); "
                        f"com $\\eta$ = {rot(hi.valor)} {'houve divergência' if (hi.divergiu > 0 or not np.isfinite(hi.val_media)) else 'não houve divergência, mas'} "
                        f"a validação foi {pt(hi.val_media)}.")
            if param == "batch":
                lo, hi = d.iloc[0], d.iloc[-1]
                return (f"Com batch {rot(lo.valor)} o treino levou {pt(lo.tempo_s, 1)} s (muitas atualizações por época) e com {rot(hi.valor)} "
                        f"{pt(hi.tempo_s, 1)} s; a validação foi {pt(lo.val_media)} e {pt(hi.val_media)}, respectivamente, e a melhor época "
                        f"passou de {pt(lo.epoca_parada, 0)} para {pt(hi.epoca_parada, 0)}.")
            if param in ("weight_decay", "dropout"):
                lo, hi = d.iloc[0], d.iloc[-1]
                return (f"Do menor ({rot(lo.valor)}) para o maior valor ({rot(hi.valor)}), o desempenho de treino foi de {pt(lo.treino_media)} para "
                        f"{pt(hi.treino_media)} e o gap de {pt(lo.gap)} para {pt(hi.gap)}: a hipótese de redução da variância "
                        f"{'se confirmou' if hi.gap < lo.gap else 'não se confirmou'} (diagnóstico com o maior valor: {hi.diagnostico}).")
            if param == "loss":
                if p == "clf":
                    a, w = self.get(param, "BCE"), self.get(param, "BCE_pos_weight")
                    if a is not None and w is not None:
                        return (f"Com limiar fixo de 0,5, a BCE sem peso obteve F1 macro {pt(a.val_media)} contra {pt(w.val_media)} com "
                                f"\\texttt{{pos\\_weight}}; o peso {'ajudou' if w.val_media > a.val_media else 'não ajudou'} a classe minoritária.")
                else:
                    return ("Como a métrica principal é o RMSE, a perda MSE (que otimiza o mesmo critério) é favorecida; MAE e Huber "
                            "reduzem o peso dos resíduos grandes (outliers de pressão).")
            if param == "scale":
                a, s_ = self.get(param, "True"), self.get(param, "False")
                if a is not None and s_ is not None:
                    return (f"Sem padronização a validação foi {pt(s_.val_media)} contra {pt(a.val_media)} com padronização"
                            f"{' (divergiu)' if s_.divergiu > 0 else ''}: a hipótese {'se confirmou' if pior(s_.val_media, a.val_media) or not np.isfinite(s_.val_media) else 'não se confirmou'}.")
        except Exception as e:  # nunca quebrar o relatório por causa do texto automático
            return f"% verificação automática falhou: {esc(e)}"
        return ""


HIPOTESE = {
    "neurons": "Poucos neurônios dão baixa capacidade (alto viés, underfitting); muitos neurônios reduzem o viés, mas aumentam a variância e o gap entre treino e validação.",
    "layers": "Mais camadas aumentam a capacidade e o número de parâmetros, mas tornam a otimização mais difícil (gradientes atravessam mais derivadas encadeadas); para dados tabulares com sinal fraco, redes rasas devem bastar.",
    "activation": "Sigmoid e Tanh saturam (derivada próxima de zero para entradas grandes) e devem convergir mais devagar; ReLU, LeakyReLU e GELU não saturam para entradas positivas.",
    "optimizer": "SGD sem momento com $\\eta=10^{-3}$ deve convergir lentamente; momento acelera a descida em direções consistentes, e RMSprop/Adam/AdamW adaptam o passo por parâmetro e convergem mais rápido.",
    "lr": "$\\eta$ muito alto faz a atualização $\\theta \\leftarrow \\theta - \\eta\\nabla_\\theta L$ ultrapassar o mínimo (oscilação ou divergência); $\\eta$ muito baixo faz a perda cair devagar e o treino pode terminar antes de convergir.",
    "batch": "Batches pequenos dão gradientes ruidosos e muitas atualizações por época (mais tempo por época, algum efeito regularizador); batches grandes dão gradientes estáveis, mas poucas atualizações por época, exigindo mais épocas ou $\\eta$ maior.",
    "weight_decay": "A penalização L2 encolhe os pesos, reduz a variância e o gap; valores altos demais causam underfitting.",
    "dropout": "Desligar neurônios aleatoriamente durante o treino é uma regularização estocástica: reduz o desempenho de treino e o gap; em excesso, causa underfitting. Na inferência o dropout é desativado com \\texttt{eval()}.",
    "loss": "Regressão: MSE penaliza resíduos grandes quadraticamente (sensível a outliers), MAE linearmente e Huber combina os dois. Classificação: com 17\\% de positivos, a BCE sem peso tende a prever a classe majoritária; \\texttt{pos\\_weight} $= n_{neg}/n_{pos}$ equilibra o gradiente das duas classes.",
    "scale": "Sem padronização, features em escalas muito diferentes (ex.: \\texttt{Income\\_USD} na casa de $10^5$ e variáveis one-hot em $\\{0,1\\}$) produzem gradientes desbalanceados e saturação, prejudicando ou impedindo o treino.",
}
MECANISMO = {
    "neurons": "O número de parâmetros cresce aproximadamente com o quadrado dos neurônios por camada; mais parâmetros permitem memorizar ruído do treino (variância), o que aparece como aumento do gap. Com sinal fraco (como neste conjunto sintético), a capacidade extra rende pouco na validação.",
    "layers": "Cada camada a mais compõe uma transformação não linear; o gradiente das primeiras camadas é produto de várias derivadas, podendo encolher. O early stopping limita o overfitting, mas não compensa a otimização mais difícil.",
    "activation": "A derivada da sigmoide é no máximo 0,25 e a da tanh no máximo 1, ambas tendendo a zero na saturação; ao multiplicar essas derivadas entre camadas, o gradiente enfraquece. A ReLU tem derivada 1 para entradas positivas (mas zero para negativas, o que pode \\emph{matar} neurônios), e a LeakyReLU/GELU mantêm gradiente também para entradas negativas.",
    "optimizer": "O SGD usa o mesmo passo para todos os parâmetros; o momento acumula uma média dos gradientes anteriores; RMSprop e Adam dividem o passo pela raiz da média dos gradientes ao quadrado, adaptando-o a cada parâmetro; o AdamW aplica o decaimento de pesos de forma desacoplada (aqui com o padrão do PyTorch, 0,01).",
    "lr": "O tamanho do passo controla o equilíbrio entre velocidade e estabilidade: passos grandes amplificam o erro de cada atualização (a perda explode e vira NaN), passos pequenos precisam de muitas épocas e o limite de épocas é atingido antes da convergência.",
    "batch": "Com $n$ amostras de treino e batch $B$, há $\\lceil n/B \\rceil$ atualizações por época: o batch pequeno faz mais atualizações (mais tempo por época, pois cada passo tem custo fixo) com gradiente ruidoso; o batch grande aproveita melhor o paralelismo, mas dá poucos passos por época.",
    "weight_decay": "O termo $\\lambda\\lVert\\theta\\rVert^2$ soma $2\\lambda\\theta$ ao gradiente, puxando os pesos para zero e suavizando a função aprendida (menos variância, mais viés).",
    "dropout": "Cada passo treina uma sub-rede diferente, o que impede a coadaptação de neurônios e funciona como um conjunto (ensemble) implícito; o desempenho de treino cai porque a rede é avaliada com ruído durante o ajuste.",
    "loss": "A perda define a direção do gradiente: no MSE, o gradiente de cada amostra é proporcional ao resíduo; no MAE é constante em módulo; o Huber é quadrático perto de zero e linear longe. Na BCE com \\texttt{pos\\_weight}, o gradiente das amostras positivas é multiplicado pelo peso.",
    "scale": "A descida do gradiente usa o mesmo $\\eta$ para todos os pesos; features com variância enorme dominam o produto $w^\\top x$, saturam as ativações e tornam a superfície de perda mal condicionada.",
}


def secao_hp(E):
    s = []
    for param in C.STUDY:
        s.append(f"\\subsection{{{C.NOMES[param]}}}\n")
        s.append(f"Valores testados: {', '.join(rot(v) for v in (C.STUDY[param]['clf'] if isinstance(C.STUDY[param], dict) else C.STUDY[param]))}"
                 f"{' (classificação) e ' + ', '.join(C.STUDY[param]['reg']) + ' (regressão)' if isinstance(C.STUDY[param], dict) else ''}"
                 f"{' (opcional)' if param not in C.OBRIGATORIOS else ''}.\n\n")
        for p in PROBS:
            s.append(E[p].tabela(param))
        for p in PROBS:
            s.append(figura(f"hp_{p}_{param}", f"{C.NOMES[param]} -- {NP[p]}: métrica final por valor (esq.) e perda de validação por época (dir., semente 1)."))
        s.append(f"\\paragraph{{(a) Hipótese.}} {HIPOTESE[param]}\n\n")
        s.append(f"\\paragraph{{(b) Evidência.}} {E['clf'].evidencia(param)} {E['reg'].evidencia(param)}\n\n")
        s.append(f"\\paragraph{{(c) Mecanismo.}} {MECANISMO[param]}\n% REVISAR: relacione o mecanismo com as curvas acima.\n\n")
        s.append(f"\\paragraph{{(d) Conclusão prática.}} {E['clf'].conclusao(param)} {E['reg'].conclusao(param)}\n% REVISAR\n\n")
    return "".join(s)


def secao_heatmap(p):
    M = T(f"heatmap_{p}_media", index_col=0); S = T(f"heatmap_{p}_desvio", index_col=0)
    linhas = []
    for b in M.columns:
        col = M[b]
        if col.notna().any():
            best = col.idxmax() if maior(p) else col.idxmin()
            linhas.append((b, best, col[best]))
    melhores = {round(float(l[1]), 8) for l in linhas}
    texto = (f"Na {NP[p]}, o melhor $\\eta$ para cada batch foi: "
             + "; ".join(f"batch {b}: $\\eta$ = {rot(lr)} ({pt(v)})" for b, lr, v in linhas) + ". ")
    if len(melhores) > 1:
        texto += ("O $\\eta$ ótimo muda com o batch, ou seja, há interação: com batches maiores há menos atualizações por época, e um "
                  "passo maior compensa (regra de escala linear $\\eta \\propto B$); com batches pequenos o gradiente é ruidoso e um "
                  "$\\eta$ alto amplifica esse ruído. ")
    else:
        texto += ("O mesmo $\\eta$ foi o melhor para todos os batches; a interação aparece apenas na intensidade do efeito "
                  "(compare as linhas do mapa), não na posição do ótimo. ")
    cab = ["$\\eta$ \\textbackslash{} batch"] + [str(c) for c in M.columns]
    rows = [[rot(i)] + [pm(M.loc[i, c], S.loc[i, c]) for c in M.columns] for i in M.index]
    return (tabela(cab, rows, f"Mapa de calor ({NP[p]}): {MET[p]} de validação, média $\\pm$ desvio.", f"tab:hm_{p}")
            + figura(f"heatmap_{p}", f"Interação taxa de aprendizado $\\times$ batch -- {NP[p]}.", "0.6") + texto + "\n\n")


def secao_busca(p):
    b = T(f"busca_aleatoria_{p}")
    cab = ["id", "neurônios", "camadas", "ativação", "otimizador", "$\\eta$", "batch", "w. decay", "dropout", f"{MET[p]} val.", "gap"]
    rows = [[int(r.valor), int(r.neurons), int(r.layers), r.activation, esc(r.optimizer), f"{r.lr:.1e}", r.batch, f"{r.weight_decay:g}",
             f"{r.dropout:g}", pm(r.val_media, r.val_desvio), pt(r.gap)] for r in b.head(8).itertuples()]
    return tabela(cab, rows, f"Busca aleatória ({C.N_RANDOM} configurações, {len(C.SEEDS)} sementes cada) -- {NP[p]}: 8 melhores na validação.", f"tab:busca_{p}")


def cfg_txt(c):
    return (f"{c['layers']} camada(s) oculta(s) de {c['neurons']} neurônios, {c['activation']}, {esc(c['optimizer'])}, "
            f"$\\eta$ = {c['lr']:.1e}, batch {c['batch']}, weight decay {c['weight_decay']:g}, dropout {c['dropout']:g}, "
            f"até {c['epochs']} épocas com paciência {c['patience']}")


# ---------------------------------------------------------------- resultados
def tabela_metricas(p):
    m = T(f"metricas_{p}")
    cols = (["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "Precision_pos", "Recall_pos", "F1_pos", "ROC_AUC", "PR_AUC"]
            if p == "clf" else ["MAE", "MSE", "RMSE", "MAPE_%", "R2", "MedAE"])
    rows = [[r["modelo"], r["conjunto"]] + [pt(r[c]) for c in cols] for _, r in m.iterrows()]
    return tabela(["modelo", "conjunto"] + [esc(c) for c in cols], rows,
                  f"Métricas da Aula 08 ({NP[p]}), calculadas com o modelo \\emph{{carregado do arquivo .pt}}."
                  + (" \\emph{pos} = classe Yes; limiar escolhido na validação." if p == "clf" else " Erros em mmHg; MAPE em \\%."),
                  f"tab:met_{p}", "ll" + "r" * len(cols))


def resumo_final(p, F):
    m = T(f"metricas_{p}"); g = T(f"generalizacao_{p}"); e = T(f"eficiencia_{p}")
    pr = C.PRIM[p]
    linhas = []
    for nome in ["Baseline RN", "RN final"]:
        mm = m[m.modelo == nome].set_index("conjunto")
        gg = g[g.modelo == nome].iloc[0]; ee = e[e.modelo == nome].iloc[0]
        cfg = F["baseline"] if nome == "Baseline RN" else F["config_final"]
        linhas.append({"problema": p, "modelo": nome, "config": cfg_txt(cfg).replace("$\\eta$", "lr").replace("\\_", "_"),
                       f"{pr}_treino": mm.loc["treino", pr], f"{pr}_validacao": mm.loc["validacao", pr], f"{pr}_teste": mm.loc["teste", pr],
                       "gap_treino_teste": gg.gap_treino_teste, "diagnostico": gg.diagnostico,
                       "cv_media": F["cv"]["media"] if nome == "RN final" else np.nan, "cv_desvio": F["cv"]["desvio"] if nome == "RN final" else np.nan,
                       "n_params": ee.n_params, "tempo_treino_s": ee.tempo_treino_s, "inf_lote_ms_por_amostra": ee.inf_lote_ms_por_amostra,
                       "tamanho_disco_kb": ee.tamanho_disco_kb, "CO2_treino_g": ee.CO2_treino_g})
    return pd.DataFrame(linhas)


def secao_resultados(p, F, E):
    pr = C.PRIM[p]
    s = [f"\\subsection{{Desempenho -- {NP[p]}}}\n", tabela_metricas(p)]
    m = T(f"metricas_{p}"); te = m[m.conjunto == "teste"].set_index("modelo")
    if p == "clf":
        lim = F["limiar"]
        s.append(figura("matrizes_confusao_clf", "Matrizes de confusão no teste.", "0.75"))
        s.append(figura("roc_pr_clf", "Curvas ROC e Precision-Recall no teste (probabilidades da rede, \\texttt{torch.sigmoid})."))
        s.append(figura("limiar_clf", "Precision, Recall e F1 macro em função do limiar, na validação (modelo final).", "0.6"))
        cte = T("limiar_clf_final")
        r05 = cte.iloc[(cte.limiar - 0.5).abs().argmin()]; rl = cte.iloc[(cte.limiar - lim["RN final"]).abs().argmin()]
        prev = float(np.mean(pd.read_csv(os.path.join(ROOT, "data", "processed", "clf_train.csv")).Hypertension == "Yes"))
        s.append(f"\\paragraph{{Limiar de decisão.}} O limiar foi escolhido na \\emph{{validação}}, maximizando o F1 macro: "
                 f"{pt(lim['RN final'], 2)} no modelo final e {pt(lim['Baseline RN'], 2)} no baseline. Na validação, com limiar 0,5 a "
                 f"Precision/Recall da classe Yes foram {pt(r05.precision_pos)}/{pt(r05.recall_pos)}, e com o limiar escolhido "
                 f"{pt(rl.precision_pos)}/{pt(rl.recall_pos)}: subir o limiar troca Recall por Precision. "
                 f"Como só {pt(100 * prev, 1)}\\% dos casos são positivos, um classificador que sempre responde \\emph{{No}} já teria "
                 f"Accuracy de {pt(1 - prev)}, sem detectar nenhum hipertenso; por isso a Accuracy é enganosa aqui. "
                 f"\\textbf{{Métrica principal:}} F1 macro (equilibra as duas classes), acompanhada de Recall da classe Yes, já que um falso "
                 f"negativo (hipertenso não identificado) custa mais que um falso positivo (que só gera um exame de confirmação). "
                 f"ROC-AUC e PR-AUC no teste: {pt(te.loc['RN final', 'ROC_AUC'])} e {pt(te.loc['RN final', 'PR_AUC'])} "
                 f"(PR-AUC de um classificador aleatório = prevalência $\\approx$ {pt(prev)}).\n\n")
    else:
        s.append(figura("real_vs_previsto_residuos_reg", "Real $\\times$ previsto e resíduos $e_i = y_i - \\hat y_i$ no teste, em mmHg."))
        s.append(f"\\paragraph{{Métrica principal e resíduos.}} Usamos o RMSE (em mmHg, mesma unidade do target) como métrica principal "
                 f"porque erros grandes de pressão são clinicamente mais graves e o RMSE os penaliza mais; o MAE "
                 f"({pt(te.loc['RN final', 'MAE'])} mmHg no teste) é reportado junto por ser menos sensível a outliers. A diferença "
                 f"RMSE $-$ MAE ({pt(te.loc['RN final', 'RMSE'] - te.loc['RN final', 'MAE'])}) indica a presença de resíduos grandes. "
                 f"R$^2$ no teste = {pt(te.loc['RN final', 'R2'])}. % REVISAR: descreva o padrão dos resíduos (faixas, viés para valores extremos).\n\n")

    # generalização
    g = T(f"generalizacao_{p}"); cv = T(f"cv_{p}")
    s.append(f"\\subsection{{Generalização -- {NP[p]}}}\n")
    rows = [[r.modelo, pt(r[f"{pr}_treino"]), pt(r[f"{pr}_validacao"]), pt(r[f"{pr}_teste"]), pt(r.gap_treino_teste),
             pt(100 * r.gap_relativo, 1) + "\\%", int(r.epoca_melhor), r.diagnostico] for _, r in g.iterrows()]
    s.append(tabela(["modelo", "treino", "validação", "teste", "gap", "gap rel.", "época parada", "diagnóstico"], rows,
                    f"{MET[p]} e gap de generalização (gap = treino $-$ teste{'' if p == 'clf' else '; para erro, teste $-$ treino'}). "
                    "Overfitting se gap relativo $>$ 10\\%; underfitting se o treino fica a menos de 0,05 do modelo constante.", f"tab:gen_{p}"))
    s.append(figura(f"curvas_aprendizado_{p}", f"Curvas de aprendizado (perda por época) e ponto do early stopping -- {NP[p]}."))
    s.append(tabela(["fold"] + [esc(c) for c in cv.columns[1:]], [[int(r.fold)] + [pt(v) for v in list(r)[2:]] for r in cv.itertuples()]
                    + [["média $\\pm$ desvio"] + [pm(cv[c].mean(), cv[c].std()) for c in cv.columns[1:]]],
                    f"Validação cruzada {C.K_FOLDS}-fold da configuração final no conjunto de treino ({NP[p]}).", f"tab:cv_{p}"))
    # extremos do estudo
    ext = []
    for param in ["neurons", "layers", "lr", "weight_decay", "dropout"]:
        d = E.d(param)
        for r in (d.iloc[0], d.iloc[-1]):
            ext.append([C.NOMES[param], rot(r.valor), pm(r.treino_media, r.treino_desvio), pm(r.val_media, r.val_desvio), pt(r.gap), r.diagnostico])
    s.append(tabela(["hiperparâmetro", "valor", "treino", "validação", "gap", "diagnóstico"], ext,
                    f"Configurações extremas do estudo ({NP[p]}): classificação em under/overfitting.", f"tab:ext_{p}"))
    gf = g[g.modelo == "RN final"].iloc[0]
    s.append(f"O modelo final tem gap de {pt(gf.gap_treino_teste)} ({pt(100 * gf.gap_relativo, 1)}\\%) e foi classificado como "
             f"\\emph{{{gf.diagnostico}}}; na validação cruzada, {MET[p]} = {pm(F['cv']['media'], F['cv']['desvio'])}. "
             f"\\textbf{{Viés e variância:}} aumentar neurônios/camadas reduziu o desempenho de treino perdido (menos viés) e, "
             f"quando o gap cresceu, aumentou a variância; weight decay e dropout fazem o caminho inverso. % REVISAR com os números das Tabelas.\n\n")

    # eficiência
    e = T(f"eficiencia_{p}")
    s.append(f"\\subsection{{Eficiência computacional -- {NP[p]}}}\n")
    cols = [("tempo_treino_s", "treino (s)", 2), ("epocas_rodadas", "épocas", 0), ("pico_rss_treino_mb", "RAM pico treino (MB)", 0),
            ("inf_lote_ms_por_amostra", "inf. lote (ms/amostra)", 5), ("inf_lote_vazao_s", "vazão lote (amostras/s)", 0),
            ("inf_individual_ms", "inf. individual (ms)", 4), ("inf_com_preproc_ms_por_amostra", "c/ pré-proc. (ms/amostra)", 4),
            ("n_params", "parâmetros", 0), ("MACs_por_amostra", "MACs/amostra", 0), ("tamanho_disco_kb", "disco (KB)", 1),
            ("energia_treino_kWh", "energia treino (kWh)", 7), ("CO2_treino_g", "CO$_2$ treino (g)", 5)]
    rows = [[c[1]] + [pt(r[c[0]], c[2]) for _, r in e.iterrows()] for c in cols]
    s.append(tabela(["medida"] + list(e.modelo), rows, f"Eficiência ({NP[p]}). Energia estimada como potência assumida "
                    f"({C.POTENCIA_ASSUMIDA_W:g} W) $\\times$ tempo; CO$_2$ = kWh $\\times$ 45,8 g/kWh (fator do Brasil, Aula 08).", f"tab:ef_{p}"))
    ef = e[e.modelo == "RN final"].iloc[0]
    s.append(f"A inferência em lote ({pt(ef.inf_lote_ms_por_amostra, 5)} ms/amostra) é muito mais rápida por amostra que a individual "
             f"({pt(ef.inf_individual_ms, 4)} ms), pois o custo fixo de cada chamada (despacho do Python e do PyTorch) é dividido entre "
             f"todas as amostras e a multiplicação de matrizes aproveita a vetorização da CPU.\n\n")
    return "".join(s)


def custo_hp(E):
    s = []
    for p in PROBS:
        dn, dl, db = E[p].d("neurons"), E[p].d("layers"), E[p].d("batch")
        s.append(f"Na {NP[p]}: de {rot(dn.iloc[0].valor)} para {rot(dn.iloc[-1].valor)} neurônios, os parâmetros foram de "
                 f"{int(dn.iloc[0].n_params)} para {int(dn.iloc[-1].n_params)} (arquivo $\\approx$ parâmetros $\\times$ 4 bytes) e o tempo médio "
                 f"de {pt(dn.iloc[0].tempo_s, 1)} s para {pt(dn.iloc[-1].tempo_s, 1)} s; de {rot(dl.iloc[0].valor)} para {rot(dl.iloc[-1].valor)} camadas, "
                 f"de {int(dl.iloc[0].n_params)} para {int(dl.iloc[-1].n_params)} parâmetros; o batch não altera os parâmetros, mas o tempo foi de "
                 f"{pt(db.iloc[0].tempo_s, 1)} s (batch {rot(db.iloc[0].valor)}) a {pt(db.iloc[-1].tempo_s, 1)} s ({rot(db.iloc[-1].valor)}). ")
    return "".join(s)


def secao_persistencia(F):
    s = [r"""\subsection{state\_dict $\times$ modelo inteiro}
Há duas formas de salvar uma rede em PyTorch. (i) \texttt{torch.save(modelo.state\_dict(), caminho)} grava só os tensores de pesos e
vieses; para carregar, recria-se a arquitetura (\texttt{MLP(...)}) e chama-se \texttt{load\_state\_dict}. Não depende da estrutura de
pastas do código e pode ser lido com \texttt{torch.load(..., weights\_only=True)}, que só aceita tensores e tipos simples e por isso
não executa código arbitrário. (ii) \texttt{torch.save(modelo, caminho)} serializa o objeto inteiro com \emph{pickle}: fica acoplado
ao nome da classe e ao caminho do módulo (\texttt{src.rn\_models.MLP}), quebra se o código mudar de lugar e exige
\texttt{weights\_only=False}, o que é inseguro com arquivos desconhecidos (um pickle malicioso executa código ao ser carregado).
\textbf{Adotamos o state\_dict}, dentro de um checkpoint completo.
"""]
    rows = []
    for p in PROBS:
        ps = F[p]["persistencia"]
        rows.append([NP[p], pt(ps["state_dict_kb"], 1), pt(ps["tamanho_inteiro_kb"], 1), esc(ps["weights_only_true"]),
                     int(ps["n_params"]), pt(ps["bytes_float32_teorico_kb"], 1), pt(ps["state_dict_kb"] - ps["bytes_float32_teorico_kb"], 1)])
    s.append(tabela(["problema", "checkpoint (KB)", "modelo inteiro (KB)", "inteiro c/ weights\\_only=True", "parâmetros",
                     "parâm. $\\times$ 4 B (KB)", "overhead (KB)"], rows,
                    "Tamanho em disco e relação com o número de parâmetros (float32). O overhead do checkpoint inclui média/desvio do "
                    "scaler, nomes das features, codificação das categóricas e metadados do formato.", "tab:tam"))
    s.append(r"""\subsection{Conteúdo do checkpoint}
Cada arquivo (\texttt{models/modelo\_classificacao.pt} e \texttt{models/modelo\_regressao.pt}) é um dicionário com:
\texttt{state\_dict}; \texttt{arch} (dimensão de entrada, camadas ocultas, ativação, dropout -- para recriar a rede);
\texttt{config\_treino} (todos os hiperparâmetros); \texttt{x\_media} e \texttt{x\_desvio} (StandardScaler ajustado no treino);
\texttt{features} (nomes das colunas na ordem de entrada); \texttt{encoder} (medianas, modas e categorias do treino, para imputar e
codificar amostras novas a partir das colunas originais); \texttt{y\_media}/\texttt{y\_desvio} (target padronizado, regressão);
\texttt{classes} e \texttt{limiar} (classificação). Sem o pré-processamento, o modelo carregado receberia entradas em outra escala e
produziria previsões erradas.

\subsection{Carregamento em ambiente limpo e fidelidade}
O script \texttt{src/inference.py} é executado em um \emph{processo Python separado} (não compartilha nenhuma variável do treino):
carrega o arquivo com \texttt{map\_location="cpu"} e \texttt{weights\_only=True}, recria a arquitetura, aplica o pré-processamento
salvo, chama \texttt{modelo.eval()} e prevê dentro de \texttt{torch.inference\_mode()}. Saída obtida (classificação):
""")
    li = F["clf"]["inferencia_ambiente_limpo"]
    s.append("\\begin{verbatim}\n$ " + li["comando"] + "\n" + "\n".join(li["stdout"].splitlines()[:12]) + "\n\\end{verbatim}\n")
    rows = []
    for p in PROBS:
        fd = T(f"fidelidade_{p}")
        pr = C.PRIM[p]
        for r in fd.itertuples():
            rows.append([NP[p], r.modelo, "sim" if r.allclose else "não", f"{r.dif_max_abs:.2e}", pt(getattr(r, f"{pr}_teste_antes_de_salvar"), 6),
                         pt(getattr(r, f"{pr}_teste_modelo_carregado"), 6), "sim" if r.metricas_identicas else "não"])
        li = F[p]["inferencia_ambiente_limpo"]
        rows.append([NP[p], "processo separado (20 amostras)", "--", f"{li.get('dif_max_abs_vs_memoria', float('nan')):.2e}", "--", "--", "--"])
    s.append(tabela(["problema", "modelo", "torch.allclose", "dif. máx. abs.", "métrica antes de salvar", "métrica do carregado", "idênticas"],
                    rows, "Verificação de fidelidade no teste: saídas do modelo em memória $\\times$ modelo carregado do .pt (métrica principal).", "tab:fid"))
    rows = []
    for p in PROBS:
        de = F[p]["demo_eval"]
        rows.append([NP[p], esc(de["origem"]), pt(de["val_com_eval"]), pm(de["val_sem_eval_media"], de["val_sem_eval_desvio"]),
                     f"{de['dif_max_entre_passadas']:.3f}"])
    s.append(r"""\paragraph{E se esquecer o \texttt{eval()}?} Em modo de treino o dropout continua sorteando neurônios a cada chamada: a mesma
entrada gera saídas diferentes e o desempenho cai, porque a rede é avaliada com ruído (e, se houvesse BatchNorm, usaria as estatísticas
do lote em vez das acumuladas). \texttt{eval()} desliga o dropout; \texttt{torch.no\_grad()}/\texttt{inference\_mode()} apenas evitam
guardar o grafo do autograd (menos memória e tempo), não substituem o \texttt{eval()}.
""")
    s.append(tabela(["problema", "modelo usado", "com eval()", "sem eval() (5 passadas)", "dif. máx. entre 2 passadas"], rows,
                    "Efeito de esquecer \\texttt{eval()} (métrica principal na validação).", "tab:eval"))
    s.append(r"""\paragraph{Portabilidade.} O treino foi feito em CPU. Se fosse feito em GPU, os tensores do \texttt{state\_dict} ficariam
salvos como tensores CUDA; \texttt{map\_location="cpu"} os remapeia para a CPU ao carregar, permitindo usar o modelo em uma máquina sem
GPU (sem ele, o carregamento falha com erro de dispositivo CUDA).

""")
    rows = []
    for p in PROBS:
        ex = F[p]["extras"]
        for nome, r in ex.items():
            if "erro" in r:
                rows.append([NP[p], esc(nome), "--", "--", "--", esc(r["erro"])])
            else:
                rows.append([NP[p], esc(nome), pt(r["tamanho_kb"], 1) if r.get("tamanho_kb") is not None else pt(F[p]["persistencia"]["state_dict_kb"], 1),
                             pt(r["tempo_por_amostra_ms"], 5), pt(r["vazao_amostras_s"], 0), pt(r[C.PRIM[p]])])
    s.append(tabela(["problema", "versão", "disco (KB)", "ms/amostra (lote)", "amostras/s", "métrica principal (teste)"], rows,
                    "Desafio opcional: TorchScript (\\texttt{torch.jit.trace}) e quantização dinâmica int8 (\\texttt{quantize\\_dynamic}).", "tab:extras"))
    return "".join(s)


def respostas(p, F, E):
    pr = C.PRIM[p]
    sens = E.sens
    top = sens.head(3)
    g = T(f"generalizacao_{p}").set_index("modelo").loc["RN final"]
    fd = T(f"fidelidade_{p}").set_index("modelo").loc["RN final"]
    ef = T(f"eficiencia_{p}").set_index("modelo").loc["RN final"]
    m = T(f"metricas_{p}"); te = m[m.conjunto == "teste"].set_index("modelo")
    s = [f"\\subsection{{Respostas -- {NP[p]}}}\n\\begin{{enumerate}}[label=(\\alph*)]\n"]
    s.append("\\item " + "; ".join(f"{r.hiperparametro} (amplitude {pt(r.amplitude)}, {'efeito real' if r.efeito_real else 'dentro do ruído'})"
                                   for r in top.itertuples())
             + ". Os mecanismos estão na Seção~\\ref{sec:hp}. Hiperparâmetros com efeito desprezível (dentro do desvio entre sementes): "
             + (", ".join(r.hiperparametro for r in sens[~sens.efeito_real].itertuples()) or "nenhum") + ". % REVISAR\n")
    s.append(f"\\item Ver Tabela~\\ref{{tab:hm_{p}}} e o texto que a acompanha.\n")
    s.append(f"\\item {MET[p]} treino/validação/teste = {pt(g[f'{pr}_treino'])}/{pt(g[f'{pr}_validacao'])}/{pt(g[f'{pr}_teste'])}; "
             f"gap = {pt(g.gap_treino_teste)} ({pt(100 * g.gap_relativo, 1)}\\%), diagnóstico: \\emph{{{g.diagnostico}}}; "
             f"K-fold: {pm(F['cv']['media'], F['cv']['desvio'])}.\n")
    s.append(f"\\item {'Sim' if fd.metricas_identicas else 'Não exatamente'}: diferença máxima absoluta entre as saídas = {fd.dif_max_abs:.2e} "
             f"(\\texttt{{torch.allclose}} = {fd.allclose}). Cuidados: salvar média/desvio do scaler (e do target), a codificação das "
             f"categóricas e a ordem das features; recriar a mesma arquitetura a partir de \\texttt{{arch}}; chamar \\texttt{{eval()}}; "
             f"usar \\texttt{{map\\_location}} e \\texttt{{weights\\_only=True}}.\n")
    s.append(f"\\item {int(ef.n_params)} parâmetros, {int(ef.MACs_por_amostra)} MACs por amostra, {pt(ef.tamanho_disco_kb, 1)} KB em disco, "
             f"{pt(ef.tempo_treino_s, 1)} s de treino e {pt(ef.inf_lote_ms_por_amostra, 5)} ms por amostra em lote: custo desprezível para "
             f"uso em clínica, em servidor ou em dispositivo de borda. % REVISAR conforme o contexto do cliente\n")
    comp_path = f"{C.TABLES}/comparacao_{p}.csv"
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path).set_index("modelo")
        ant = comp.drop(index=["Baseline RN", "RN final"], errors="ignore")
        best_ant = ant[pr].idxmax() if maior(p) else ant[pr].idxmin()
        rn = comp.loc["RN final", pr]; va = ant.loc[best_ant, pr]
        melhor_rn = rn > va if maior(p) else rn < va
        s.append(f"\\item No teste, a rede final obteve {MET[p]} = {pt(rn)}, contra {pt(va)} do melhor modelo da atividade anterior "
                 f"({best_ant}). " + ("A rede foi melhor, " if melhor_rn else "A rede não superou os modelos anteriores, ")
                 + f"com diferença de {pt(abs(rn - va))}"
                 + ("; considerando o custo extra de ajuste e a menor interpretabilidade, a recomendação depende de a diferença ser relevante para o cliente."
                    if melhor_rn else "; como árvores e ensembles são mais simples de ajustar e de interpretar, \\textbf{não recomendamos} a rede para este problema.")
                 + " % REVISAR\n")
    else:
        s.append("\\item TODO (aluno): recomendação final.\n")
    s.append("\\end{enumerate}\n")
    return "".join(s)


# ---------------------------------------------------------------- documento
def main():
    E = {p: Estudo(p) for p in PROBS}
    F = {p: J(f"final_{p}") for p in PROBS}
    pd.concat([resumo_final(p, F[p]) for p in PROBS]).to_csv(f"{C.TABLES}/resumo_final.csv", index=False)
    amb = F["clf"]["ambiente"]
    n = {p: (len(pd.read_csv(os.path.join(ROOT, "data", "processed", f"{p}_train.csv"))),
             len(pd.read_csv(os.path.join(ROOT, "data", "processed", f"{p}_test.csv")))) for p in PROBS}
    ndim = {p: len(Encoder(p).fit(split_val(p)[0]).feature_names) for p in PROBS}
    B = C.BASELINE
    figs_rel = os.path.relpath(C.FIGS, os.path.join(ROOT, "docs")).replace("\\", "/")

    sens_rows = []
    for p in PROBS:
        for r in E[p].sens.itertuples():
            sens_rows.append([NP[p], r.hiperparametro, rot(r.melhor_valor), pt(r.melhor_val), rot(r.pior_valor), pt(r.pior_val),
                              pt(r.amplitude), pt(r.soma_desvios), "sim" if r.efeito_real else "não"])

    res = pd.read_csv(f"{C.TABLES}/resumo_final.csv")
    res_rows = [[NP[r.problema], r.modelo, pt(getattr(r, f"{C.PRIM[r.problema]}_treino")), pt(getattr(r, f"{C.PRIM[r.problema]}_validacao")),
                 pt(getattr(r, f"{C.PRIM[r.problema]}_teste")), pt(r.gap_treino_teste), r.diagnostico,
                 pm(r.cv_media, r.cv_desvio) if np.isfinite(r.cv_media) else "--", int(r.n_params), pt(r.tempo_treino_s, 1),
                 pt(r.inf_lote_ms_por_amostra, 5), pt(r.tamanho_disco_kb, 1)] for r in res.itertuples()]

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}\usepackage[T1]{{fontenc}}\usepackage[brazilian]{{babel}}
\usepackage{{graphicx,booktabs,float,hyperref,geometry,amsmath,amssymb,enumitem}}\geometry{{margin=2.2cm}}
\graphicspath{{{{{figs_rel}/}}}}
\begin{{document}}
\begin{{titlepage}}\centering
{{\large UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE\\Departamento de Computação e Tecnologia\\DCT1401 -- Inteligência Artificial (2026.2)\\Professor: Thommas K. S. Flores\\}}\vspace{{2cm}}
{{\LARGE\bfseries Redes Neurais Artificiais com PyTorch:\\Classificação e Regressão\par}}
\vspace{{2cm}}Aluno: Jonas Jeronimo Cirilo Silva \quad Matrícula: 20240026090\\[1ex]
Dataset: \url{{https://www.kaggle.com/datasets/mobeenfatimah/human-digital-twin-dataset-for-ai-and-healthcare}}\\[1ex]
Repositório: \url{{https://github.com/xonazz/IA_thomas2}}
\end{{titlepage}}
\tableofcontents\newpage

\section{{Introdução}}
Este relatório avalia se redes neurais do tipo perceptron multicamadas (MLP), implementadas em PyTorch, são uma alternativa viável
aos modelos da atividade anterior e, principalmente, como os hiperparâmetros afetam o desempenho, a generalização e o custo.
Usamos o mesmo conjunto \emph{{Human Digital Twin}} (Kaggle, arquivo \texttt{{digital\_twin\_profile.csv}}: 5000 indivíduos sintéticos e
30 colunas), os mesmos targets e as mesmas features:
\textbf{{Classificação:}} \texttt{{Hypertension}} (Yes/No; 17,2\% de Yes), 24 features, sem as colunas de pressão arterial e sem diagnósticos
(a pressão define a hipertensão). \textbf{{Regressão:}} \texttt{{Blood\_Pressure\_Systolic}} (mmHg), 26 features (as mesmas, mais os
diagnósticos \texttt{{Hypertension}} e \texttt{{Diabetes}}; sem a pressão diastólica). Após a codificação, a entrada da rede tem
{ndim['clf']} colunas na
classificação e {ndim['reg']} na regressão (12 numéricas, one-hot das categóricas e 3 ordinais).

\section{{Protocolo Experimental}}
\begin{{itemize}}
\item \textbf{{Divisão:}} a mesma da atividade anterior: 80\% treino / 20\% teste, \texttt{{random\_state=42}}, estratificada na classificação
({n['clf'][0]} treino e {n['clf'][1]} teste). Dentro do treino, 20\% viram validação (estratificada na classificação): 3200 amostras para
ajuste e 800 para validação. A validação é usada para comparar hiperparâmetros, para o early stopping e para escolher o limiar;
o teste é usado uma única vez, com o modelo final carregado do arquivo.
\item \textbf{{Pré-processamento (ajustado só no treino):}} imputação pela mediana (numéricas) e moda (categóricas), one-hot, códigos
ordinais para \texttt{{Stress\_Level}}, \texttt{{Physical\_Activity}} e \texttt{{Alcohol}}, e padronização (média e desvio do treino),
aplicada também à validação, ao teste e à inferência. Na regressão, o target também é padronizado e as previsões voltam para mmHg antes das métricas.
\item \textbf{{Tensores:}} \texttt{{torch.float32}}, \texttt{{TensorDataset}} e \texttt{{DataLoader}} (embaralhamento com gerador semeado).
\item \textbf{{Reprodutibilidade:}} sementes fixas de \texttt{{random}}, \texttt{{numpy}} e \texttt{{torch}}; cada configuração foi treinada com
{len(C.SEEDS)} sementes ({', '.join(map(str, C.SEEDS))}) e reportamos média $\pm$ desvio padrão. Todas as execuções estão em \texttt{{logs/hiperparametros.csv}}.
\item \textbf{{Ambiente:}} {esc(amb['so'])}; CPU {esc(amb['cpu'])} ({amb['nucleos_logicos']} núcleos lógicos, {amb['threads_torch']} threads do PyTorch);
{amb['ram_gb']} GB de RAM; GPU: {esc(amb['gpu'])}. Python {amb['python']}, PyTorch {esc(amb['torch'])}, NumPy {amb['numpy']},
pandas {amb['pandas']}, scikit-learn {amb['sklearn']}.
\end{{itemize}}

\section{{Redes Neurais Implementadas}}
A classe \texttt{{MLP}} (\texttt{{src/rn\_models.py}}) é parametrizável: recebe a dimensão de entrada, a lista de camadas ocultas, a
ativação (\texttt{{getattr(nn, nome)}}) e a taxa de dropout, e empilha \texttt{{Linear $\to$ ativação ($\to$ Dropout)}}, terminando
em uma camada \texttt{{Linear}} de 1 neurônio. O laço de treinamento (\texttt{{src/rn\_train.py}}) usa autograd
(\texttt{{zero\_grad}}, \texttt{{backward}}, \texttt{{step}}), mede a perda de validação a cada época e faz early stopping guardando o melhor
\texttt{{state\_dict}}; perdas NaN/infinitas interrompem o treino (divergência).
\begin{{itemize}}
\item \textbf{{Regressão:}} saída linear de 1 neurônio, sem ativação, pois a pressão é um valor real sem limite; perda
\texttt{{nn.MSELoss}}, que corresponde à máxima verossimilhança com ruído gaussiano e otimiza diretamente o RMSE.
\item \textbf{{Classificação binária:}} 1 neurônio de saída sem sigmoide (logit) e \texttt{{nn.BCEWithLogitsLoss}}, que combina sigmoide
e entropia cruzada de forma numericamente estável (\emph{{log-sum-exp}}); a probabilidade é obtida com \texttt{{torch.sigmoid}} só na
inferência e no cálculo de ROC/PR. Como há desbalanceamento (17\% de Yes), o baseline usa \texttt{{pos\_weight}} $= n_{{neg}}/n_{{pos}}$
$\approx$ 4,8; o efeito de retirá-lo é avaliado no estudo da função de perda.
\end{{itemize}}

\section{{Estudo de Hiperparâmetros}}\label{{sec:hp}}
\textbf{{Configuração de referência (baseline):}} {cfg_txt(B)}, perda MSE (regressão) ou BCE com \texttt{{pos\_weight}} (classificação),
com padronização. Cada estudo altera \emph{{um}} hiperparâmetro por vez, com as mesmas sementes e o mesmo limite de épocas.
Métrica de comparação na validação: F1 macro com limiar 0,5 (classificação) e RMSE em mmHg (regressão). Todos os seis obrigatórios
foram avaliados nos dois problemas, além de quatro opcionais (weight decay, dropout, função de perda e padronização).
Diferenças menores que a soma dos desvios entre sementes do melhor e do pior valor não são tratadas como efeito real.

{secao_hp(E)}

\subsection{{Interação entre hiperparâmetros: taxa de aprendizado $\times$ batch}}
{secao_heatmap('clf')}{secao_heatmap('reg')}

\subsection{{Busca final}}
Com base nos estudos isolados, fizemos uma busca aleatória com {C.N_RANDOM} configurações ({len(C.SEEDS)} sementes cada) em um espaço
que exclui os valores que divergiram ou foram claramente piores (ex.: $\eta \in [3\cdot10^{{-4}}, 10^{{-2}}]$ log-uniforme, sem Sigmoid
e sem SGD simples, 1 a 3 camadas). A escolha usa apenas a média na validação.
{secao_busca('clf')}{secao_busca('reg')}
\textbf{{Configuração final -- classificação:}} {cfg_txt(F['clf']['config_final'])}; validação {pm(F['clf']['busca_val_media'], F['clf']['busca_val_desvio'])}.\\
\textbf{{Configuração final -- regressão:}} {cfg_txt(F['reg']['config_final'])}; validação {pm(F['reg']['busca_val_media'], F['reg']['busca_val_desvio'])} mmHg.

\section{{Persistência e Inferência}}
{secao_persistencia(F)}

\section{{Resultados}}
{secao_resultados('clf', F['clf'], E['clf'])}
{secao_resultados('reg', F['reg'], E['reg'])}
\subsection{{Efeito dos hiperparâmetros no custo e implantação}}
{custo_hp(E)}
As redes finais têm poucos milhares de parâmetros e ocupam dezenas de KB: cabem com folga em um dispositivo de borda (ex.: Raspberry Pi
ou microcontrolador com alguns MB de RAM). Para reduzir ainda mais, é possível usar menos neurônios (o estudo mostra que a capacidade
extra rende pouco), quantização int8 (Tabela~\ref{{tab:extras}}) e poda de pesos pequenos. % REVISAR

\section{{Análise e Discussão}}
{tabela(['problema', 'modelo', 'treino', 'validação', 'teste', 'gap', 'diagnóstico', 'K-fold', 'parâm.', 'treino (s)', 'ms/amostra', 'KB'], res_rows,
        'Tabela-resumo final (métrica principal: F1 macro na classificação, RMSE em mmHg na regressão). Arquivo: \\texttt{reports/tables/rn/resumo\\_final.csv}.', 'tab:resumo')}
{tabela(['problema', 'hiperparâmetro', 'melhor', 'val.', 'pior', 'val.', 'amplitude', 'soma desvios', 'efeito real'], sens_rows,
        'Sensibilidade: hiperparâmetros ordenados pelo impacto na validação (melhor $-$ pior valor testado).', 'tab:sens')}
{respostas('clf', F['clf'], E['clf'])}
{respostas('reg', F['reg'], E['reg'])}
\subsection{{Comparação com a atividade anterior}}
""" + "".join(tabela(["modelo"] + [esc(c) for c in pd.read_csv(f"{C.TABLES}/comparacao_{p}.csv").columns[1:]],
                     [[esc(r[0])] + [pt(v) for v in r[1:]] for r in pd.read_csv(f"{C.TABLES}/comparacao_{p}.csv").itertuples(index=False)],
                     f"Teste ({NP[p]}): modelos da atividade anterior $\\times$ redes neurais, mesmos conjuntos de treino e teste. Tempos medidos em execuções diferentes.",
                     f"tab:comp_{p}") for p in PROBS if os.path.exists(f"{C.TABLES}/comparacao_{p}.csv")) + rf"""
\subsection{{Vantagens e desvantagens das redes neurais para estes dados}}
\textbf{{Vantagens:}} modelam relações não lineares e interações sem que precisemos especificá-las; o mesmo código serve para
regressão e classificação; o modelo final é pequeno e rápido na inferência. \textbf{{Desvantagens:}} são sensíveis à escala das
features (Seção~\ref{{sec:hp}}, padronização) e aos hiperparâmetros (taxa de aprendizado e batch mudam o resultado e até causam
divergência); exigem pré-processamento salvo junto com o modelo; são pouco interpretáveis (ao contrário da árvore de decisão e das regras
do sistema especialista); e, em um conjunto tabular pequeno (5000 linhas) e de sinal fraco, não há dados suficientes para a capacidade
extra fazer diferença.

\subsection{{Limitações}}
Conjunto sintético e pequeno, com sinal fraco (AUC e R$^2$ baixos para todos os modelos); uma única divisão treino/teste; {len(C.SEEDS)}
sementes por configuração; o estudo de um fator por vez não cobre todas as interações (só lr $\times$ batch foi varrido em 2D); a busca
aleatória com {C.N_RANDOM} configurações é limitada; o limiar 0,5 no estudo de hiperparâmetros (o limiar ótimo só foi escolhido para o
modelo final); a energia foi estimada por potência assumida ({C.POTENCIA_ASSUMIDA_W:g} W), não medida; os tempos dependem do hardware.

\section{{Considerações Finais}}
\textbf{{TODO (aluno):}} dificuldades encontradas, o que aprendeu sobre hiperparâmetros e sobre salvar/carregar modelos.

\section{{Referência ao Repositório}}
Código: \url{{https://github.com/xonazz/IA_thomas2}}. Estrutura: \texttt{{src/rn\_models.py}} (classe MLP), \texttt{{src/rn\_train.py}}
(treino, early stopping, sementes, métricas), \texttt{{src/rn\_prep.py}} (pré-processamento exportável), \texttt{{src/rn\_study.py}}
(estudo de hiperparâmetros), \texttt{{src/rn\_final.py}} (modelo final, persistência, métricas, generalização e eficiência),
\texttt{{src/inference.py}} (inferência independente), \texttt{{src/rn\_report.py}} (este relatório), \texttt{{run\_rn.py}} (executa tudo),
\texttt{{models/}} (arquivos .pt), \texttt{{logs/hiperparametros.csv}} (todas as execuções), \texttt{{reports/tables/rn}} e
\texttt{{reports/figures/rn}} (tabelas e gráficos) e \texttt{{notebooks/rn/}} (notebooks 03 a 08).
\end{{document}}
"""
    out = os.path.join(ROOT, "docs", "relatorio_rn_rapido.tex" if C.RAPIDO else "relatorio_rn.tex")
    open(out, "w", encoding="utf-8").write(tex)
    print("relatório gerado em", out)


if __name__ == "__main__":
    main()
