"""Configuração central: caminhos, sementes, colunas e fatores."""
import os

SEED = 42
N_SPLITS = 5
N_REP_INFERENCIA = 30
FATOR_CO2_G_KWH = 45.8          # fator de emissão do Brasil (Aula 08)
POTENCIA_ASSUMIDA_W = 45.0      # potência média da CPU (ALTERE para a sua máquina)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed")
RULES = os.path.join(ROOT, "rules")
TABLES = os.path.join(ROOT, "reports", "tables")
FIGS = os.path.join(ROOT, "reports", "figures")
MODELS = os.path.join(ROOT, "models")
LOGS = os.path.join(ROOT, "logs")
for d in (TABLES, FIGS, MODELS, LOGS):
    os.makedirs(d, exist_ok=True)

TARGETS = {"reg": "Blood_Pressure_Systolic", "clf": "Hypertension"}
CLASSES = {"clf": ["No", "Yes"]}          # classe positiva = "Yes"

# Colunas por tipo de tratamento
NUM = ["Age", "Height_cm", "Weight_kg", "BMI", "Income_USD", "Heart_Rate", "Body_Temperature",
       "Oxygen_Saturation", "Respiratory_Rate", "Daily_Water_Intake_L", "Screen_Time_Hours", "Sleep_Hours"]
CAT_BASE = ["Gender", "Blood_Group", "Occupation", "Marital_Status", "Education", "Country",
            "City_Type", "Smoking", "Family_History"]
ORD_COLS = ["Stress_Level", "Physical_Activity", "Alcohol"]
ORD_ORDER = [["Low", "Moderate", "High"],
             ["Sedentary", "Light", "Moderate", "Active"],
             ["Never", "Occasionally", "Frequently"]]

# Classificação: sem colunas de pressão arterial e sem diagnósticos (evita vazamento: a pressão define a hipertensão)
# Regressão: Hypertension e Diabetes entram como features (diagnósticos registrados à parte); sem pressão diastólica
CAT = {"clf": CAT_BASE, "reg": CAT_BASE + ["Hypertension", "Diabetes"]}
FEATURES = {p: NUM + CAT[p] + ORD_COLS for p in ("clf", "reg")}
