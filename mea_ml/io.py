"""Result paths and consistency checks."""

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import joblib

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_protocol(bundle_path):
    from . import training
    return {
        "version": 2,
        "settings": {"RANDOM_STATE": training.RANDOM_STATE,
            "OUTER_N_SPLITS": training.OUTER_N_SPLITS, "N_REPEATS": training.N_REPEATS,
            "OUTER_SEEDS": training.OUTER_SEEDS, "CORR_THRESHOLD": training.CORR_THRESHOLD,
            "PRESERVE_CORRELATED_FEATURES": sorted(training.PRESERVE_CORRELATED_FEATURES)},
        "tuning_budget": {"n_init": training.BAYES_N_INIT, "n_iter": training.BAYES_N_ITER},
        "training_source_sha256": sha256(Path(training.__file__)),
        "training_bundle_sha256": sha256(bundle_path),
        "versions": {name: version(name) for name in ["numpy", "pandas", "scipy",
            "scikit-learn", "xgboost", "catboost", "joblib"]},
    }


def load_ml_summary():
    folder = RESULTS / "03_ml"
    if (folder / "export_in_progress.json").exists():
        raise RuntimeError("Incomplete training export. Run notebook 3 again.")
    summary = json.loads((folder / "ml_summary.json").read_text())
    for name, expected in summary["export_sha256"].items():
        if sha256(folder / name) != expected:
            raise RuntimeError(f"Mismatched training output: {name}. Run notebook 3 again.")
    if sha256(RESULTS / "01_dataset/training_bundle.joblib") != summary["training_bundle_sha256"]:
        raise RuntimeError("Dataset changed after training. Run notebook 3 again.")
    return summary


def load_analysis():
    summary = load_ml_summary()
    bundle_path = RESULTS / "01_dataset/training_bundle.joblib"
    artifact_path = RESULTS / "03_ml/model_artifacts.joblib"
    bundle = joblib.load(bundle_path)
    artifact = joblib.load(artifact_path)
    protocol = training_protocol(bundle_path)
    if artifact["training_protocol"] != protocol:
        raise RuntimeError("Training code or environment changed. Run notebook 3 again.")
    if joblib.hash(protocol) != artifact["training_protocol_hash"]:
        raise RuntimeError("Inconsistent training metadata. Run notebook 3 again.")
    lineage = {
        "training_bundle_sha256": summary["training_bundle_sha256"],
        "training_artifact_sha256": summary["export_sha256"]["model_artifacts.joblib"],
        "selected_model": artifact["selected_model_name"],
        "selected_feature_set": artifact["selected_feature_set"],
        "n_repeats": artifact["n_repeats"], "outer_n_splits": artifact["outer_n_splits"],
        "training_random_state": artifact["random_state"],
        "training_protocol_hash": artifact["training_protocol_hash"],
    }
    return bundle, artifact, lineage
