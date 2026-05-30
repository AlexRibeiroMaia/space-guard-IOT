# 🛰️ Space Guard

Sistema de monitoramento de queimadas com dados orbitais do INPE, classificação por Machine Learning e análise via IA Generativa (Claude).

> Projeto desenvolvido para a **Global Solution 2024 — FIAP**

---

## Visão Geral

O Space Guard consome focos de incêndio em tempo real do INPE, classifica cada foco em **Alto / Médio / Baixo** risco usando um modelo RandomForest treinado com dados do INPE Queimadas, e oferece um assistente de IA (Claude Sonnet) para análise contextual dos dados.

```
Dados INPE (TerraBrasilis) → API Flask → Modelo ML → Frontend + Chat IA
```

---

## Arquitetura

```
space-guard/
├── data/
│   ├── gerar_dataset.py        # Gerador do dataset de treino (5000 registros)
│   └── focos_inpe_treino.csv   # Dataset gerado (não versionado em produção)
│
├── model/
│   ├── treinar_modelo.py       # Treinamento do RandomForestClassifier
│   ├── modelo_risco.pkl        # Modelo treinado (joblib)
│   ├── encoder_bioma.pkl       # LabelEncoder para biomas
│   └── modelo_meta.json        # Metadados: acurácia, features, classes
│
├── api/
│   └── api.py                  # Backend Flask (4 endpoints)
│
├── frontend/
│   └── index.html              # SPA completa (HTML + CSS + JS)
│
├── .env                        # Chaves de API (não commitar)
├── requirements.txt
└── README.md
```

---

## Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Backend | Python 3.10+, Flask 3.x, Flask-CORS |
| Machine Learning | scikit-learn (RandomForestClassifier), pandas, joblib |
| IA Generativa | Anthropic Claude Sonnet (`claude-sonnet-4-6`) |
| Dados | INPE TerraBrasilis (focos ativos do dia) |
| Frontend | HTML5 + CSS3 + JavaScript puro (sem frameworks) |

---

## Modelo de Machine Learning

- **Algoritmo:** RandomForestClassifier
- **Features:** `frp`, `brightness`, `confidence`, `biome_enc`, `month`, `hour`
- **Classes:** Alto / Médio / Baixo
- **Acurácia:** ~93% (cross-validation 5-fold)
- **Parâmetros:** 150 estimadores, max_depth=10, class_weight=balanced

### Distribuição do dataset de treino

| Classe | Proporção |
|--------|-----------|
| Baixo  | ~53%      |
| Médio  | ~40%      |
| Alto   | ~7%       |

---

## Fonte de Dados INPE

O sistema busca focos ativos do dia via:

```
https://terrabrasilis.dpi.inpe.br/geoserver/ams1h/ows
  ?service=WFS&version=1.0.0&request=GetFeature
  &typeName=ams1h:active-fire-today&outputFormat=csv
```

- Dados atualizados ao longo do dia pelos satélites do INPE
- Cache local de 10 minutos para não sobrecarregar o servidor
- Fallback automático para dados simulados se o INPE estiver indisponível
- O badge **"✅ INPE Tempo Real"** ou **"⚠️ Dados Simulados"** indica a fonte atual

> **Sazonalidade:** O período crítico de queimadas no Brasil é **julho a outubro** (estação seca). Em outros meses, o número de focos de risco Alto é naturalmente baixo — o modelo reflete isso.

---

## Pré-requisitos

- Python 3.10 ou superior
- Chave da API Anthropic → [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

---

## Instalação e Execução

### 1. Clone e instale as dependências

```bash
cd space-guard
pip3 install -r requirements.txt
```

### 2. Configure a chave da API

Edite o arquivo `.env` na raiz do projeto:

```env
ANTHROPIC_API_KEY=sk-ant-...sua-chave-aqui...
```

### 3. Gere o dataset (opcional — já gerado)

```bash
python3 data/gerar_dataset.py
```

### 4. Treine o modelo (opcional — já treinado)

```bash
python3 model/treinar_modelo.py
```

### 5. Inicie a API

```bash
python3 api/api.py
```

A API sobe em `http://127.0.0.1:5000`.

### 6. Abra o frontend

Abra o arquivo `frontend/index.html` no browser:

```bash
open frontend/index.html
# ou via Live Server no VS Code (porta 5500)
```

---

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/status` | Status da API e metadados do modelo ML |
| GET | `/focos` | Focos ativos do INPE classificados pelo modelo |
| POST | `/classificar` | Classifica um foco enviado no body JSON |
| POST | `/chat` | Chat com Claude usando dados INPE como contexto |

### Exemplos

**GET /status**
```json
{
  "status": "online",
  "algorithm": "RandomForestClassifier",
  "accuracy": 0.929,
  "cv_accuracy": 0.927
}
```

**POST /classificar**
```bash
curl -X POST http://127.0.0.1:5000/classificar \
  -H "Content-Type: application/json" \
  -d '{"frp": 350, "brightness": 460, "confidence": 92, "biome": "Amazônia", "month": 8, "hour": 14}'
```

```json
{
  "risk_label": "Alto",
  "risk_score": 0.977,
  "probabilities": {"Alto": 0.977, "Médio": 0.023, "Baixo": 0.0}
}
```

**POST /chat**
```bash
curl -X POST http://127.0.0.1:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Quais biomas estão com mais risco hoje?"}'
```

---

## Funcionalidades do Frontend

- **Dashboard de métricas** — focos ativos, risco alto, risco médio, precisão do modelo
- **Lista de focos** — classificados em tempo real com badge de risco colorido
- **Indicador de fonte** — mostra se os dados são do INPE ou simulados
- **Info do modelo ML** — algoritmo, acurácia, features, número de árvores
- **Simulação de campo** — simula dispositivo móvel dentro/fora de zona de risco
- **Chat com IA** — assistente Claude com acesso aos dados INPE do dia como contexto

---

## Segurança

- Nunca commite o arquivo `.env` com sua chave real
- Adicione `.env` ao `.gitignore` antes de versionar o projeto:

```bash
echo ".env" >> .gitignore
```

---

## Equipe

Desenvolvido por:

**Alex Ribeiro Maia - RM557356**

**Igor Neris Soares Alves - RM560088**

**Guilherme Jun Conheci - RM559986**

**Alessandro da Silva Lira - RM560512**

**Leonardo Carvalho Jeronimo Santos - RM560380**

— FIAP Global Solution 2024
