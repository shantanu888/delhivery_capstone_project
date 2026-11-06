from pathlib import Path
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, build_features, clean_shipments

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "raw_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")

# Capture raw-export diagnostics after exact duplicate removal and before filling.
dedup = raw.drop_duplicates().copy()
service_spellings_before = int(dedup["service_type"].nunique())
missing_distance = int(dedup["distance_km"].isna().sum())
missing_weight = int(dedup["parcel_weight_kg"].isna().sum())
missing_declared = int(dedup["declared_value"].isna().sum())
missing_payment = int(dedup["payment_mode"].isna().sum())
sentinel_weight = int((dedup["parcel_weight_kg"] > 100).sum())

clean, medians = clean_shipments(raw)
service_spellings_after = int(clean["service_type"].nunique())
model_table = build_features(clean, hubs)
model_table["is_delayed"] = (model_table["delivery_hours"] > model_table["promised_hours"]).astype("int64")

# Ensure one row per unique shipment and no missing model inputs/target.
model_table = model_table.drop_duplicates(subset=["shipment_id"])
model_table.to_csv(DATA / "modelling_table.csv", index=False, float_format="%.4f")

hours = model_table["delivery_hours"]
q1 = float(hours.quantile(0.25))
q3 = float(hours.quantile(0.75))
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

value = model_table["declared_value_inr"]
log_value = np.log(value)
metrics = [
    ("raw_rows", len(raw)),
    ("duplicate_rows_removed", len(raw) - len(dedup)),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance),
    ("missing_parcel_weight_before", missing_weight),
    ("sentinel_weight_rows", sentinel_weight),
    ("missing_declared_value_before", missing_declared),
    ("missing_payment_mode_before", missing_payment),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(model_table)),
    ("nulls_in_features_and_target", int(model_table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(model_table["is_delayed"].sum())),
    ("delay_rate", float(model_table["is_delayed"].mean())),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower),
    ("hours_upper_bound", upper),
    ("hours_outlier_count", int(((hours < lower) | (hours > upper)).sum())),
    ("declared_value_mean_inr", float(value.mean())),
    ("declared_value_median_inr", float(value.median())),
    ("log_declared_value_mean", float(log_value.mean())),
    ("log_declared_value_median", float(log_value.median())),
]
pd.DataFrame(metrics, columns=["metric", "value"]).to_csv(OUTPUTS / "part_a_checks.csv", index=False, float_format="%.4f")

# Delay evidence by service, hub tier, and distance band.
distance_band = pd.cut(
    model_table["distance_km"], bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"], right=True, include_lowest=True,
)
rows = []
for dimension, groups in [
    ("service_type", model_table.groupby("service_type", sort=True, observed=True)),
    ("hub_tier", model_table.groupby("hub_tier", sort=True, observed=True)),
    ("distance_band", model_table.assign(distance_band=distance_band).groupby("distance_band", sort=True, observed=True)),
]:
    for group, frame in groups:
        rows.append({
            "dimension": dimension,
            "group": group,
            "shipments": int(len(frame)),
            "delayed": int(frame["is_delayed"].sum()),
            "delay_rate": float(frame["is_delayed"].mean()),
            "mean_hours": float(frame["delivery_hours"].mean()),
        })
pd.DataFrame(rows).to_csv(OUTPUTS / "part_a_delay_by_group.csv", index=False, float_format="%.4f")

numeric = model_table.select_dtypes(include=[np.number])
corr = numeric.corr(numeric_only=True)
corr.to_csv(OUTPUTS / "part_a_correlations.csv", index=True, index_label="column", float_format="%.4f")

print(f"Part A complete: {len(model_table)} modelling rows; {int(model_table['is_delayed'].sum())} delayed.")
print(f"Training medians: {medians}")
