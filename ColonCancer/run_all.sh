#!/usr/bin/env bash
# Lance un run AutoGluon **ultra-léger** tracké avec MLflow
# Usage : bash run_all.sh && mlflow ui
#
# Variantes manuelles possibles :
#   python mlflow_track.py --light                    # 20 features, LightGBM seul
#   python mlflow_track.py --light --time-limit 60    # Encore plus rapide
#   python mlflow_track.py                            # Mode normal (50 features)

set -e
cd "$(dirname "$0")"

echo "=== Run ultra-léger: --light --preset medium_quality --time-limit 120 ==="
python mlflow_track.py --light --preset medium_quality --time-limit 120

echo ""
echo "✅ Run terminé."
echo "Lancez:  mlflow ui --port 5000"