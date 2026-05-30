"""
Treinamento do modelo de classificação de risco de incêndio (Space Guard).
Usa RandomForestClassifier com features do INPE para predizer Alto/Médio/Baixo.
Salva modelo, encoder e metadados em model/.
"""

import sys
import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/focos_inpe_treino.csv")
MODEL_DIR = os.path.dirname(__file__)

FEATURES = ["frp", "brightness", "confidence", "biome_enc", "month", "hour"]
TARGET = "risk_label"


def barra_ascii(valor, total=50):
    preenchido = int(valor * total)
    return "[" + "#" * preenchido + "-" * (total - preenchido) + f"] {valor:.4f}"


def main():
    print("=" * 60)
    print("SPACE GUARD — Treinamento do Modelo de Risco de Incêndio")
    print("=" * 60)

    # 1. Carregar dados
    print(f"\n[1/6] Carregando dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Registros: {len(df)} | Colunas: {list(df.columns)}")

    # 2. Encodar bioma
    print("\n[2/6] Aplicando LabelEncoder no campo 'biome'")
    le = LabelEncoder()
    df["biome_enc"] = le.fit_transform(df["biome"])
    biome_classes = list(le.classes_)
    print(f"      Classes: {biome_classes}")

    # 3. Separar features e target
    X = df[FEATURES]
    y = df[TARGET]
    print(f"\n[3/6] Features: {FEATURES}")
    print(f"      Distribuição de classes:")
    for cls, cnt in y.value_counts().items():
        print(f"        {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")

    # 4. Split treino/teste com stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[4/6] Divisão: {len(X_train)} treino / {len(X_test)} teste (80/20, stratified)")

    # 5. Treinar RandomForestClassifier
    print("\n[5/6] Treinando RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("      Treinamento concluído.")

    # Cross-validation (5 folds)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_mean = float(cv_scores.mean())
    print(f"\n      Cross-validation (5-fold):")
    print(f"        Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"        Média:  {cv_mean:.4f} ({cv_mean*100:.2f}%)")

    # Avaliação no conjunto de teste
    y_pred = clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    print(f"\n      Acurácia no teste: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=sorted(y.unique())))

    # Importância das features
    print("--- Importância das Features ---")
    importances = clf.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp:
        print(f"  {feat:15s} {barra_ascii(imp)}")

    # 6. Salvar artefatos
    print("\n[6/6] Salvando artefatos em model/")

    model_path = os.path.join(MODEL_DIR, "modelo_risco.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoder_bioma.pkl")
    meta_path = os.path.join(MODEL_DIR, "modelo_meta.json")

    joblib.dump(clf, model_path)
    print(f"      Salvo: {model_path}")

    joblib.dump(le, encoder_path)
    print(f"      Salvo: {encoder_path}")

    meta = {
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
        "accuracy": round(test_acc, 4),
        "cv_accuracy": round(cv_mean, 4),
        "biome_classes": biome_classes,
        "n_estimators": 150,
        "algorithm": "RandomForestClassifier",
        "trained_on": "focos_inpe_treino.csv",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      Salvo: {meta_path}")

    # 7. Predições de teste
    print("\n--- Predições de Validação ---")
    casos_teste = pd.DataFrame([
        {"frp": 350.0, "brightness": 460.0, "confidence": 95, "biome_enc": le.transform(["Amazônia"])[0],  "month": 8,  "hour": 14},
        {"frp": 55.0,  "brightness": 345.0, "confidence": 65, "biome_enc": le.transform(["Cerrado"])[0],   "month": 5,  "hour": 10},
        {"frp": 12.0,  "brightness": 298.0, "confidence": 40, "biome_enc": le.transform(["Pampa"])[0],     "month": 2,  "hour": 22},
    ])
    descricoes = [
        "Alta intensidade, Amazônia, agosto, tarde",
        "Intensidade moderada, Cerrado, maio, manhã",
        "Baixa intensidade, Pampa, fevereiro, noite",
    ]
    probas = clf.predict_proba(casos_teste[FEATURES])
    preds = clf.predict(casos_teste[FEATURES])
    classes = clf.classes_

    for i, (desc, pred, prob) in enumerate(zip(descricoes, preds, probas)):
        print(f"\n  Caso {i+1}: {desc}")
        print(f"    Predição: {pred}")
        print(f"    Probabilidades: { {c: f'{p:.3f}' for c, p in zip(classes, prob)} }")

    print(f"\n{'='*60}")
    if test_acc >= 0.75:
        print(f"✓ Acurácia {test_acc*100:.1f}% — meta de 75% atingida.")
    else:
        print(f"⚠ Acurácia {test_acc*100:.1f}% — abaixo da meta de 75%.")
    print("=" * 60)


if __name__ == "__main__":
    main()
