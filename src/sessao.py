"""Cronômetro de sessão de elaboração de regras.
Uso: python -m src.sessao clf     (ou: python -m src.sessao reg)
ENTER inicia; ENTER de novo encerra. Pausas: digite 'p' + ENTER para pausar/retomar
(o tempo pausado não conta e a interrupção é registrada). Ao final, informe quantas
regras criou; a linha é gravada em logs/cronometro_regras.csv.
"""
import csv, sys, time, os
from datetime import datetime
from .config import LOGS

prob = sys.argv[1] if len(sys.argv) > 1 else "clf"
path = f"{LOGS}/cronometro_regras.csv"
input("ENTER para iniciar a sessão...")
inicio = datetime.now(); t0 = time.perf_counter(); pausado = 0.0; interrup = 0; tp = None
print("Cronometrando. 'p'+ENTER pausa/retoma; ENTER (vazio) encerra.")
while True:
    cmd = input().strip().lower()
    if cmd == "p":
        if tp is None:
            tp = time.perf_counter(); interrup += 1; print("PAUSADO")
        else:
            pausado += time.perf_counter() - tp; tp = None; print("RETOMADO")
    else:
        break
if tp is not None:
    pausado += time.perf_counter() - tp
dur = (time.perf_counter() - t0 - pausado) / 60
n = int(input("Quantas regras criou nesta sessão? "))
novo = not os.path.exists(path) or os.path.getsize(path) == 0
with open(path, "a", newline="") as f:
    w = csv.writer(f)
    if novo:
        w.writerow(["problema", "sessao", "data", "hora_inicio", "duracao_min", "n_regras_criadas", "interrupcoes"])
    sess = sum(1 for l in open(path)) if not novo else 1
    w.writerow([prob, sess, inicio.strftime("%Y-%m-%d"), inicio.strftime("%H:%M"), f"{dur:.2f}", n, interrup])
print(f"Registrado: {dur:.1f} min, {n} regras, {interrup} interrupções.")
