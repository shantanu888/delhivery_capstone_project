from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"
MODELS.mkdir(exist_ok=True)


def main():
    config = pd.read_csv(OUTPUTS / "part_c_final_config.csv").iloc[0]
    max_depth = None if str(config["max_depth"]).lower() == "unlimited" else int(config["max_depth"])
    params = {
        "n_estimators": int(config["n_estimators"]),
        "max_depth": max_depth,
        "min_samples_leaf": int(config["min_samples_leaf"]),
    }
    model_table = pd.read_csv(DATA / "modelling_table.csv")
    X = model_table[FEATURES]
    y = model_table["is_delayed"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
        ("model", RandomForestClassifier(random_state=42, **params)),
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODELS / "delay_model.pkl")
    pd.DataFrame({"threshold": [float(config["threshold"])]}).to_csv(
        MODELS / "delay_threshold.csv", index=False, float_format="%.4f")
    pd.DataFrame({"feature": FEATURES}).to_csv(MODELS / "feature_order.csv", index=False)

    raw = pd.read_csv(DATA / "raw_shipments.csv")
    _, medians = clean_shipments(raw)
    pd.DataFrame([{"column": key, "median": value} for key, value in medians.items()]).to_csv(
        MODELS / "impute_values.csv", index=False, float_format="%.4f")

    reloaded = joblib.load(MODELS / "delay_model.pkl")
    in_memory_prob = pipeline.predict_proba(X_test)[:, 1]
    reloaded_prob = reloaded.predict_proba(X_test)[:, 1]
    diff = float(np.max(np.abs(in_memory_prob - reloaded_prob)))
    pd.DataFrame([
        {"metric": "test_rows_scored", "value": len(X_test)},
        {"metric": "reload_max_abs_probability_diff", "value": diff},
        {"metric": "threshold", "value": float(config["threshold"])},
    ]).to_csv(OUTPUTS / "part_d_reload_check.csv", index=False, float_format="%.10f")
    assert len(X_test) == 180 and diff <= 1e-9
    print("Model packaged; test rows:", len(X_test), "reload diff:", diff)


if __name__ == "__main__":
    main()
