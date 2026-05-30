"""
Gerador de dataset simulado de focos de incêndio do INPE.
Produz um CSV com 5000 registros seguindo a estrutura do INPE Queimadas,
com coordenadas geográficas reais por bioma e distribuição de risco realista.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BIOMAS = {
    "Amazônia": {
        "estados": ["AM", "PA", "RO", "AC", "AP", "RR", "MT", "TO", "MA"],
        "lat_range": (-10.0, 2.5),
        "lon_range": (-73.0, -46.0),
    },
    "Cerrado": {
        "estados": ["GO", "DF", "MG", "BA", "MS", "MT", "TO", "PI", "MA", "SP"],
        "lat_range": (-23.0, -3.0),
        "lon_range": (-60.0, -41.0),
    },
    "Caatinga": {
        "estados": ["BA", "CE", "PE", "RN", "PB", "PI", "SE", "AL", "MA"],
        "lat_range": (-15.0, -3.0),
        "lon_range": (-45.0, -35.0),
    },
    "Mata Atlântica": {
        "estados": ["SP", "RJ", "ES", "MG", "PR", "SC", "RS", "BA"],
        "lat_range": (-29.0, -7.0),
        "lon_range": (-52.0, -34.0),
    },
    "Pantanal": {
        "estados": ["MS", "MT"],
        "lat_range": (-22.0, -15.0),
        "lon_range": (-58.5, -55.0),
    },
    "Pampa": {
        "estados": ["RS"],
        "lat_range": (-33.8, -27.0),
        "lon_range": (-57.5, -49.5),
    },
}

SATELITES = ["AQUA_M-T", "AQUA_M-D", "GOES-16", "TERRA_M-T", "TERRA_M-D"]

PESO_BIOMA = {
    "Amazônia": 1.40,
    "Cerrado": 1.00,
    "Caatinga": 0.90,
    "Mata Atlântica": 0.80,
    "Pantanal": 1.30,
    "Pampa": 0.70,
}


def calcular_score_continuo(frp, brightness, biome, month, hour):
    """Retorna score contínuo de risco (quanto maior, maior o risco)."""
    # FRP: 0-50 pontos (distribuição exponencial média 60 → maioria abaixo de 120)
    if frp > 300:
        frp_pts = 50
    elif frp > 150:
        frp_pts = 35
    elif frp > 80:
        frp_pts = 20
    elif frp > 40:
        frp_pts = 10
    else:
        frp_pts = 3

    # Brightness: 0-40 pontos (normal μ=340, σ=55)
    if brightness > 440:
        bri_pts = 40
    elif brightness > 390:
        bri_pts = 25
    elif brightness > 350:
        bri_pts = 12
    elif brightness > 320:
        bri_pts = 5
    else:
        bri_pts = 1

    base = frp_pts + bri_pts

    # Bioma: multiplicador histórico
    base *= PESO_BIOMA.get(biome, 1.0)

    # Sazonalidade: estação seca jul-out
    if 7 <= month <= 10:
        base *= 1.45
    elif month in [6, 11]:
        base *= 1.15

    # Pico de calor diurno
    if 12 <= hour <= 17:
        base *= 1.30
    elif 10 <= hour <= 19:
        base *= 1.10

    return base


def gerar_dataset(n=5000):
    registros = []
    biomas_lista = list(BIOMAS.keys())
    # Distribuição por bioma: Amazônia domina, Pampa é pequeno
    pesos_bioma_dist = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]

    for _ in range(n):
        biome = np.random.choice(biomas_lista, p=pesos_bioma_dist)
        cfg = BIOMAS[biome]

        lat = round(float(np.random.uniform(*cfg["lat_range"])), 6)
        lon = round(float(np.random.uniform(*cfg["lon_range"])), 6)
        state = random.choice(cfg["estados"])
        satellite = random.choice(SATELITES)

        # Distribuição temporal: mais focos em jul-out (estação seca)
        month_weights = [0.04, 0.04, 0.05, 0.05, 0.07, 0.07, 0.13, 0.15, 0.14, 0.12, 0.07, 0.07]
        month = int(np.random.choice(range(1, 13), p=month_weights))
        day = random.randint(1, 28)
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        datahora = datetime(2024, month, day, hour, minute).strftime("%Y-%m-%dT%H:%M:%S")

        frp = round(float(np.random.exponential(60)), 2)
        frp = max(0.5, min(frp, 2000.0))

        brightness = round(float(np.random.normal(340, 55)), 2)
        brightness = max(280.0, min(brightness, 550.0))

        confidence = random.randint(30, 100)

        score_raw = calcular_score_continuo(frp, brightness, biome, month, hour)

        registros.append({
            "datahora": datahora,
            "latitude": lat,
            "longitude": lon,
            "frp": frp,
            "brightness": brightness,
            "confidence": confidence,
            "satellite": satellite,
            "biome": biome,
            "state": state,
            "month": month,
            "hour": hour,
            "_score_raw": score_raw,
        })

    df = pd.DataFrame(registros)

    # Thresholds via percentil → garante distribuição-alvo: ~7% Alto, ~40% Médio
    p_alto = float(np.percentile(df["_score_raw"], 93))    # top 7%
    p_medio = float(np.percentile(df["_score_raw"], 53))   # próximos ~40%

    def label_from_score(s):
        if s >= p_alto:
            return "Alto"
        elif s >= p_medio:
            return "Médio"
        else:
            return "Baixo"

    df["risk_score"] = df["_score_raw"].apply(lambda s: round(min(s / (p_alto * 1.5), 1.0), 4))
    df["risk_label"] = df["_score_raw"].apply(label_from_score)
    df = df.drop(columns=["_score_raw"])

    return df


if __name__ == "__main__":
    print("Gerando dataset de focos de incêndio simulados do INPE...")
    df = gerar_dataset(5000)

    output_path = os.path.join(os.path.dirname(__file__), "focos_inpe_treino.csv")
    df.to_csv(output_path, index=False)

    print(f"\nTotal de registros gerados: {len(df)}")
    print("\nDistribuição de risco:")
    dist = df["risk_label"].value_counts()
    for label, count in dist.items():
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:6s}: {count:5d} ({pct:5.1f}%)  {bar}")

    print(f"\nArquivo salvo em: {output_path}")
    print("[DATASET GERADO COM SUCESSO]")
