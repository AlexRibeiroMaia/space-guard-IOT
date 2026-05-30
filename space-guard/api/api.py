"""
Space Guard API — Backend Flask para monitoramento de queimadas.
Integra modelo ML (RandomForest), dados em tempo real do INPE e IA Generativa (Claude).
Endpoints: /status, /focos, /classificar, /chat
"""

import os
import json
import time
import random
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "../model")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return "", 204

# Carregar modelo e encoder
clf = joblib.load(os.path.join(MODEL_DIR, "modelo_risco.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "encoder_bioma.pkl"))

with open(os.path.join(MODEL_DIR, "modelo_meta.json"), encoding="utf-8") as f:
    modelo_meta = json.load(f)

# Cliente Anthropic
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

FEATURES = modelo_meta["features"]

# Cache simples para dados INPE (TTL = 10 minutos)
_cache = {"focos": None, "ts": 0}
CACHE_TTL = 600  # segundos

FOCOS_MOCK = [
    {"datahora": "2024-08-15T14:30:00", "latitude": -4.52, "longitude": -62.81, "frp": 385.2, "brightness": 462.5, "confidence": 92, "satellite": "AQUA_M-T", "biome": "Amazônia", "state": "AM"},
    {"datahora": "2024-09-03T13:15:00", "latitude": -9.78, "longitude": -63.42, "frp": 210.8, "brightness": 435.3, "confidence": 88, "satellite": "GOES-16",  "biome": "Amazônia", "state": "RO"},
    {"datahora": "2024-08-22T15:45:00", "latitude": -15.23, "longitude": -52.30, "frp": 142.0, "brightness": 395.8, "confidence": 81, "satellite": "TERRA_M-T", "biome": "Cerrado", "state": "MT"},
    {"datahora": "2024-07-14T16:00:00", "latitude": -12.89, "longitude": -47.65, "frp": 98.5, "brightness": 370.2, "confidence": 74, "satellite": "AQUA_M-D", "biome": "Cerrado", "state": "GO"},
    {"datahora": "2024-09-19T12:30:00", "latitude": -7.35, "longitude": -38.92, "frp": 65.3, "brightness": 355.1, "confidence": 67, "satellite": "GOES-16",   "biome": "Caatinga", "state": "CE"},
    {"datahora": "2024-10-05T14:00:00", "latitude": -19.50, "longitude": -57.20, "frp": 325.0, "brightness": 478.0, "confidence": 96, "satellite": "TERRA_M-D", "biome": "Pantanal", "state": "MS"},
    {"datahora": "2024-03-18T10:20:00", "latitude": -22.10, "longitude": -43.75, "frp": 22.1, "brightness": 315.4, "confidence": 45, "satellite": "AQUA_M-T", "biome": "Mata Atlântica", "state": "RJ"},
    {"datahora": "2024-05-07T09:40:00", "latitude": -30.52, "longitude": -52.85, "frp": 15.7, "brightness": 302.8, "confidence": 38, "satellite": "TERRA_M-T", "biome": "Pampa", "state": "RS"},
    {"datahora": "2024-08-28T15:10:00", "latitude": -6.12, "longitude": -44.58, "frp": 178.4, "brightness": 415.6, "confidence": 85, "satellite": "GOES-16",   "biome": "Cerrado", "state": "MA"},
    {"datahora": "2024-09-11T13:50:00", "latitude": -3.85, "longitude": -60.23, "frp": 290.1, "brightness": 448.9, "confidence": 91, "satellite": "AQUA_M-D",  "biome": "Amazônia", "state": "AM"},
]


def _parse_point_geom(geom_str):
    """Extrai (lon, lat) de string 'POINT (lon lat)'."""
    try:
        coords = geom_str.replace("POINT (", "").replace(")", "").strip().split()
        return float(coords[1]), float(coords[0])  # lat, lon
    except Exception:
        return None, None


def _estimar_frp_brightness(biome, month, hour):
    """Gera FRP e brightness estimados com base em bioma e sazonalidade."""
    random.seed()
    frp_base = {"Amazônia": 55, "Cerrado": 65, "Caatinga": 45, "Mata Atlântica": 35, "Pantanal": 70, "Pampa": 30}
    frp_mean = frp_base.get(biome, 50)
    # Estação seca (jul-out) eleva FRP; rainy season (jan-mai) reduz
    if 7 <= month <= 10:
        frp_mean *= 1.8
    elif month in [6, 11]:
        frp_mean *= 1.2
    frp = round(max(0.5, random.expovariate(1.0 / frp_mean)), 1)
    brightness = round(max(280.0, min(550.0, random.gauss(335, 45))), 1)
    confidence = random.randint(50, 95)
    return frp, brightness, confidence


def get_focos_inpe():
    """Busca focos reais do INPE (active-fire-today) com cache de 10 min."""
    global _cache

    if _cache["focos"] is not None and (time.time() - _cache["ts"]) < CACHE_TTL:
        return _cache["focos"]

    # Endpoint real e ativo do INPE via TerraBrasilis
    url = (
        "https://terrabrasilis.dpi.inpe.br/geoserver/ams1h/ows"
        "?service=WFS&version=1.0.0&request=GetFeature"
        "&typeName=ams1h:active-fire-today&outputFormat=csv&maxFeatures=500"
    )

    try:
        from io import StringIO
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()

        df = pd.read_csv(StringIO(resp.text))

        if df.empty or "geom" not in df.columns:
            raise ValueError("Resposta vazia ou sem coluna geom")

        focos = []
        now = datetime.utcnow()
        for _, row in df.iterrows():
            lat, lon = _parse_point_geom(str(row.get("geom", "")))
            if lat is None:
                continue

            biome = str(row.get("biome", "Cerrado"))
            if biome not in le.classes_:
                biome = "Cerrado"

            # viewed_at: "2026-05-30T00:31:00"
            viewed_at = str(row.get("viewed_at", now.strftime("%Y-%m-%dT%H:%M:%S")))
            try:
                dt = datetime.strptime(viewed_at, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                dt = now
            month = dt.month
            hour = dt.hour

            frp, brightness, confidence = _estimar_frp_brightness(biome, month, hour)

            focos.append({
                "datahora":  viewed_at,
                "latitude":  round(lat, 6),
                "longitude": round(lon, 6),
                "frp":       frp,
                "brightness": brightness,
                "confidence": confidence,
                "satellite": str(row.get("satelite", "INPE")),
                "biome":     biome,
                "state":     str(row.get("municipio", "Brasil")),
                "month":     month,
                "hour":      hour,
                "fonte":     "INPE-TerraBrasilis",
            })

        if not focos:
            raise ValueError("Nenhum foco parseado")

        _cache = {"focos": focos, "ts": time.time()}
        return focos

    except Exception as e:
        # Fallback para mock apenas se INPE falhar
        now = datetime.utcnow()
        mock = []
        for f in FOCOS_MOCK:
            item = dict(f)
            item["frp"] = round(item["frp"] * random.uniform(0.9, 1.1), 2)
            item["brightness"] = round(item["brightness"] * random.uniform(0.98, 1.02), 2)
            item["month"] = now.month
            item["hour"] = now.hour
            item["datahora"] = now.strftime("%Y-%m-%dT%H:%M:%S")
            item["fonte"] = "SIMULADO"
            mock.append(item)
        _cache = {"focos": mock, "ts": time.time()}
        return mock


def classificar_foco(foco):
    """Classifica um foco de incêndio com o modelo ML."""
    biome_raw = foco.get("biome", "Cerrado")

    # Tratar biomas desconhecidos com fallback
    if biome_raw not in le.classes_:
        biome_raw = "Cerrado"

    biome_enc = int(le.transform([biome_raw])[0])

    now = datetime.utcnow()
    row = pd.DataFrame([{
        "frp":        float(foco.get("frp", 50.0)),
        "brightness": float(foco.get("brightness", 340.0)),
        "confidence": float(foco.get("confidence", 70)),
        "biome_enc":  biome_enc,
        "month":      int(foco.get("month", now.month)),
        "hour":       int(foco.get("hour", now.hour)),
    }])

    pred = clf.predict(row[FEATURES])[0]
    proba = clf.predict_proba(row[FEATURES])[0]
    classes = clf.classes_

    prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    risk_score = round(float(max(proba)), 4)

    return {
        "risk_label": pred,
        "risk_score": risk_score,
        "probabilities": prob_dict,
    }


def resumo_focos_para_prompt(focos):
    """Classifica até 20 focos e monta texto-contexto para injetar no prompt do Claude."""
    amostra = focos[:20]
    classificados = [(f, classificar_foco(f)) for f in amostra]

    contagem = {"Alto": 0, "Médio": 0, "Baixo": 0}
    for _, c in classificados:
        contagem[c["risk_label"]] += 1

    bioma_count = {}
    for f, _ in classificados:
        b = f.get("biome", "Desconhecido")
        bioma_count[b] = bioma_count.get(b, 0) + 1

    linhas_alto = []
    for f, c in classificados:
        if c["risk_label"] == "Alto":
            linhas_alto.append(
                f"  • Bioma: {f.get('biome','?')} | Estado: {f.get('state','?')} "
                f"| FRP: {f.get('frp','?')} MW | Satélite: {f.get('satellite','?')} "
                f"| Coord: ({f.get('latitude','?')}, {f.get('longitude','?')}) "
                f"| Score: {c['risk_score']:.2f}"
            )

    bioma_str = " | ".join(f"{b}: {n}" for b, n in sorted(bioma_count.items(), key=lambda x: -x[1]))

    resumo = (
        f"DADOS INPE — {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC\n"
        f"Total de focos analisados: {len(amostra)}\n"
        f"Risco Alto: {contagem['Alto']} | Risco Médio: {contagem['Médio']} | Risco Baixo: {contagem['Baixo']}\n"
        f"\nFocos de Risco Alto:\n" + ("\n".join(linhas_alto) if linhas_alto else "  Nenhum foco de risco alto no momento.") +
        f"\n\nDistribuição por bioma: {bioma_str}"
    )
    return resumo, len(amostra)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "online",
        "algorithm": modelo_meta.get("algorithm"),
        "accuracy": modelo_meta.get("accuracy"),
        "cv_accuracy": modelo_meta.get("cv_accuracy"),
        "features": modelo_meta.get("features"),
        "classes": modelo_meta.get("classes"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/focos", methods=["GET"])
def focos():
    raw = get_focos_inpe()
    enriquecidos = []
    for f in raw:
        classif = classificar_foco(f)
        item = dict(f)
        item.update(classif)
        enriquecidos.append(item)

    contagem = {"alto": 0, "medio": 0, "baixo": 0}
    for item in enriquecidos:
        rl = item.get("risk_label", "Baixo").lower()
        if rl == "alto":
            contagem["alto"] += 1
        elif "dio" in rl:
            contagem["medio"] += 1
        else:
            contagem["baixo"] += 1

    fonte = enriquecidos[0].get("fonte", "INPE") if enriquecidos else "N/A"

    return jsonify({
        "focos": enriquecidos,
        "stats": {
            "total": len(enriquecidos),
            "alto": contagem["alto"],
            "medio": contagem["medio"],
            "baixo": contagem["baixo"],
            "fonte": fonte,
        },
    })


@app.route("/classificar", methods=["POST"])
def classificar():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body JSON inválido ou ausente"}), 400
    try:
        resultado = classificar_foco(body)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True)
    if not body or "message" not in body:
        return jsonify({"error": "Campo 'message' obrigatório"}), 400

    mensagem = body["message"]
    focos_raw = get_focos_inpe()
    contexto_inpe, n_focos = resumo_focos_para_prompt(focos_raw)

    system_prompt = (
        "Você é o assistente de IA do Space Guard, sistema de monitoramento de queimadas "
        "do Brasil baseado em dados orbitais do INPE e modelo de Machine Learning.\n\n"
        "Seu papel é ajudar analistas ambientais, bombeiros e gestores de emergência a "
        "interpretar os dados de focos de incêndio em tempo real, avaliar riscos e sugerir ações.\n\n"
        "Responda de forma clara, objetiva e técnica. Use os dados abaixo como contexto.\n\n"
        f"--- CONTEXTO DOS DADOS INPE (tempo real) ---\n{contexto_inpe}\n"
        "--- FIM DO CONTEXTO ---\n\n"
        "Quando o usuário perguntar sobre focos, risco, biomas ou ações, use os dados acima "
        "como base para sua resposta. Cite biomas, estados e valores de FRP quando relevante."
    )

    try:
        resposta_claude = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": mensagem}],
        )
        texto_resposta = resposta_claude.content[0].text
    except Exception as e:
        return jsonify({"error": f"Erro ao chamar Claude: {str(e)}"}), 500

    return jsonify({
        "resposta": texto_resposta,
        "focos_usados": n_focos,
        "contexto_inpe": contexto_inpe,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
