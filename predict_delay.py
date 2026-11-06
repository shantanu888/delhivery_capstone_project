from pathlib import Path
import pandas as pd
import joblib

from src.shipment_prep import FEATURES, clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS, MODELS = ROOT / "data", ROOT / "outputs", ROOT / "models"


def main():
    raw = pd.read_csv(DATA / "new_shipments.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    model = joblib.load(MODELS / "delay_model.pkl")
    threshold = float(pd.read_csv(MODELS / "delay_threshold.csv").iloc[0]["threshold"])
    feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
    medians_frame = pd.read_csv(MODELS / "impute_values.csv")
    medians = dict(zip(medians_frame["column"], medians_frame["median"]))
    segments = pd.read_csv(OUTPUTS / "hub_segments.csv")[["hub_id", "segment"]]

    clean, _ = clean_shipments(raw, medians=medians)
    features = build_features(clean, hubs)
    X = features[feature_order]
    probabilities = model.predict_proba(X)[:, 1]
    scored = pd.DataFrame({
        "shipment_id": clean["shipment_id"].values,
        "hub_id": clean["hub_id"].values,
        "delay_probability": probabilities.round(4),
    })
    scored["flagged"] = (probabilities >= threshold).astype(int)
    scored = scored.merge(segments, on="hub_id", how="left", validate="many_to_one", sort=False)
    scored["action"] = "No action"
    flagged = scored["flagged"].eq(1)
    scored.loc[flagged, "action"] = "Proactive customer call"
    scored.loc[flagged & scored["segment"].eq("Escalate"), "action"] = "Escalate to hub manager"
    result = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    if result["shipment_id"].tolist() != raw["shipment_id"].tolist():
        raise AssertionError("Scoring changed the input shipment order")
    if result.isna().any().any() or result["shipment_id"].duplicated().any():
        raise AssertionError("Scoring output contains missing or duplicate shipment records")
    result.to_csv(OUTPUTS / "new_shipment_actions.csv", index=False, float_format="%.4f")
    print("Scored", len(result), "new shipments; flagged", int(result["flagged"].sum()),
          "hub-manager escalations", int(result["action"].eq("Escalate to hub manager").sum()))


if __name__ == "__main__":
    main()
