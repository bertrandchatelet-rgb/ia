#!/usr/bin/env python3
"""
MLflow + AutoGluon — Tracking d'expériences de classification RNA.

Usage:
    python mlflow_track.py                          # Config par défaut
    python mlflow_track.py --light                  # Mode ultra-léger (petite config)
    python mlflow_track.py --preset best_quality --time-limit 600
    mlflow ui                                       # Lancer l'interface web

Prérequis :
    pip install mlflow autogluon pandas scikit-learn
"""

import argparse
import os
import warnings

# DÉSACTIVER RAY AVANT TOUT IMPORT (sinon OOM)
os.environ["AG_DISABLE_RAY"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import mlflow
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# ─── Config ────────────────────────────────────────────────────────────────
DATA_ROOT = "/mnt/g/ngs/data"
PROJ = "PRJNA755688-stage12"
RANDOM_STATE = 222
TEST_SIZE = 0.20

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Backend SQLite (plus de warning)
MLFLOW_EXPERIMENT = "colon-cancer-rna"   # Nom de l'expérience

# Paramètres ajustables selon le mode
N_TOP_FEATURES_DEFAULT = 50
N_TOP_FEATURES_LIGHT = 20
EXPR_THRESHOLD_DEFAULT = 0.20
EXPR_THRESHOLD_LIGHT = 0.30


def load_data(light: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Charge, filtre et équilibre les données.

    En mode light :
      - seuil d'expression relevé (30% → moins de features)
      - top 20 features au lieu de 50
    """
    n_top = N_TOP_FEATURES_LIGHT if light else N_TOP_FEATURES_DEFAULT
    expr_threshold = EXPR_THRESHOLD_LIGHT if light else EXPR_THRESHOLD_DEFAULT

    path = f"{DATA_ROOT}/{PROJ}"
    f_list = f"{path}/liste-files.csv"
    src_dir = f"{path}/rna/"

    encoder = LabelEncoder()
    df_tot = pd.DataFrame()

    with open(f_list) as csvfile:
        import csv
        for row in csv.reader(csvfile, delimiter=","):
            sample, label = row[0], row[1]
            f1 = f"{src_dir}/results-{sample}.txt"
            if os.path.isfile(f1):
                df = pd.read_csv(f1, sep="\t")
                df["sample"] = sample
                df_pivot = df.pivot_table(
                    values=["count"], columns="RNA", index=["sample"]
                )
                if "I" in label:
                    label = "II"
                if "healthy" in label or "Normal" in label:
                    label = "control"
                df_pivot["label"] = label
                df_tot = pd.concat([df_tot, df_pivot])

    df_tot = df_tot.fillna(0)

    # Split AVANT équilibrage
    labels = df_tot["label"].copy()
    idx_train, idx_test = train_test_split(
        df_tot.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=labels,
    )
    df_train_all = df_tot.loc[idx_train].copy()
    y_train_all = labels.loc[idx_train]
    X_test_final = df_tot.loc[idx_test].reset_index(drop=True)
    y_test_final = labels.loc[idx_test].reset_index(drop=True)

    # Undersampling sur train uniquement
    class_counts = y_train_all.value_counts()
    min_count = class_counts.min()
    df_balanced = pd.DataFrame()
    for cls in class_counts.index:
        df_cls = df_train_all[y_train_all == cls]
        if len(df_cls) > min_count:
            df_cls = df_cls.sample(n=min_count, random_state=42)
        df_balanced = pd.concat([df_balanced, df_cls])

    target = encoder.fit_transform(df_balanced["label"])
    # Retire la colonne label avant le filtrage numérique
    df_balanced = df_balanced.drop(columns=["label"]).reset_index(drop=True)

    # Filtrage : expression > seuil + top N variance
    nonzero = (df_balanced > 0).sum(axis=0) / df_balanced.shape[0]
    mask = nonzero >= expr_threshold
    df_f1 = df_balanced.loc[:, mask]
    top = df_f1.var(axis=0).sort_values(ascending=False).head(n_top).index
    X_train = df_f1[top].copy()
    y_train = target

    # Test set : aligner les colonnes avec le train
    cols = [c for c in top if c in X_test_final.columns]
    X_test = X_test_final[cols].copy()
    y_test = encoder.transform(y_test_final.values)

    # Aplatir les MultiIndex
    for df_proc in [X_train, X_test]:
        if isinstance(df_proc.columns, pd.MultiIndex):
            df_proc.columns = ["_".join(str(c) for c in col) for col in df_proc.columns]

    return X_train, X_test, y_train, y_test, encoder


def train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
    preset: str = "medium_quality",
    time_limit: int = 300,
    light: bool = False,
) -> None:
    """Entraîne un modèle AutoGluon et logue tout dans MLflow.

    En mode light :
      - hyperparameters limités à LightGBM (le plus rapide)
      - num_bag_folds = 2 (minimum)
    """
    df_train = X_train.copy()
    df_train["label"] = encoder.inverse_transform(y_train)
    df_test = X_test.copy()
    df_test["label"] = encoder.inverse_transform(y_test)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"ag_{preset}_t{time_limit}"
    if light:
        run_name += "_light"
    with mlflow.start_run(run_name=run_name):
        # Log des paramètres
        mlflow.log_param("preset", preset)
        mlflow.log_param("time_limit", time_limit)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("light_mode", light)

        # AutoML (MLflow autolog intégré)
        mlflow.autolog(log_models=True, silent=True)

        # Configuration allégée pour petite config
        fit_kwargs = dict(
            train_data=df_train,
            time_limit=time_limit,
            presets=preset,
            dynamic_stacking=False,
            num_stack_levels=0,
        )

        if light:
            # GBM (LightGBM) uniquement — le modèle le plus rapide d'AutoGluon
            fit_kwargs["hyperparameters"] = {
                "GBM": {},
            }
            # Pas de bagging (plus rapide mais moins robuste)
            fit_kwargs["num_bag_folds"] = 2

        predictor = TabularPredictor(
            label="label",
            problem_type="multiclass",
            eval_metric="accuracy",
        ).fit(**fit_kwargs)

        # Évaluation sur vrai test set
        perf = predictor.evaluate(df_test, auxiliary_metrics=True, silent=True)
        for k, v in perf.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)

        leaderboard = predictor.leaderboard(df_test, silent=True)
        leaderboard_path = "leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow.log_artifact(leaderboard_path)

        # Log du chemin du modèle
        mlflow.log_param("model_path", predictor.path)

        print(f"\n[MLflow] Run: {mlflow.active_run().info.run_id}")
        print(f"[MLflow] Interface: mlflow ui --port 5000")
        print(f"[MLflow] Experiment: {MLFLOW_EXPERIMENT}")
        print(f"[OK] Preset={preset}, time_limit={time_limit}, light={light}")
        for k, v in perf.items():
            print(f"  {k}: {v:.4f}")

    mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="MLflow + AutoGluon RNA")
    parser.add_argument("--preset", default="medium_quality",
                        choices=["medium_quality", "high_quality", "best_quality"])
    parser.add_argument("--time-limit", type=int, default=300)
    parser.add_argument("--light", action="store_true",
                        help="Mode ultra-léger : moins de features, LightGBM uniquement")
    args = parser.parse_args()

    print("[1/3] Chargement des données...")
    X_train, X_test, y_train, y_test, encoder = load_data(light=args.light)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("[2/3] Entraînement AutoGluon...")
    train(X_train, X_test, y_train, y_test, encoder,
          preset=args.preset, time_limit=args.time_limit, light=args.light)

    print("[3/3] Terminé.")
    print(f"  -> mlflow ui --port 5000 (dans {MLFLOW_TRACKING_URI}/)")


if __name__ == "__main__":
    main()