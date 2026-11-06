from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]


def separation_ratio(matrix, labels):
    distances = pairwise_distances(matrix, metric="euclidean")
    across, within = [], []
    for i, j in combinations(range(len(labels)), 2):
        if labels[i] == labels[j]:
            within.append(distances[i, j])
        else:
            across.append(distances[i, j])
    return float(np.mean(across) / np.mean(within)), across, within


def main():
    data = pd.read_csv(DATA / "modelling_table.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    profile = (data.groupby("hub_id", sort=True)
        .agg(shipments=("shipment_id", "count"),
             mean_delivery_hours=("delivery_hours", "mean"),
             delay_rate=("is_delayed", "mean"),
             mean_distance_km=("distance_km", "mean"),
             mean_n_stops=("n_stops", "mean"))
        .reset_index().sort_values("hub_id", kind="stable").reset_index(drop=True))
    profile.to_csv(OUTPUTS / "hub_profile.csv", index=False, float_format="%.4f")

    # The profile is deliberately in ascending hub_id order before scaling/clustering.
    matrix = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    separation_rows = []
    labels_by_k = {}
    for k in [2, 3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        labels_by_k[k] = labels
        ratio, across, within = separation_ratio(matrix, labels)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        separation_rows.append({"K": k, "across_over_within": ratio,
                                "cluster_sizes": "-".join(map(str, sizes))})
    pd.DataFrame(separation_rows).to_csv(OUTPUTS / "part_d_separation.csv", index=False, float_format="%.4f")

    labels = labels_by_k[3]
    cluster_delay = {label: float(profile.loc[labels == label, "delay_rate"].mean())
                     for label in np.unique(labels)}
    ordered = sorted(cluster_delay, key=lambda label: cluster_delay[label])
    names = {ordered[0]: "Standard", ordered[1]: "Watch", ordered[2]: "Escalate"}
    hub_segments = profile[["hub_id"]].copy()
    hub_segments["segment"] = [names[label] for label in labels]
    hub_segments = hub_segments.merge(hubs[["hub_id", "hub_city", "hub_tier"]],
                                      on="hub_id", how="left", validate="one_to_one")
    hub_segments.to_csv(OUTPUTS / "hub_segments.csv", index=False)

    segment_profile = (profile.assign(segment=[names[label] for label in labels])
        .groupby("segment", sort=False)
        .agg(hubs=("hub_id", "count"),
             mean_delivery_hours=("mean_delivery_hours", "mean"),
             delay_rate=("delay_rate", "mean"),
             mean_distance_km=("mean_distance_km", "mean"),
             mean_n_stops=("mean_n_stops", "mean"))
        .reindex(["Standard", "Watch", "Escalate"]).reset_index())
    segment_profile.to_csv(OUTPUTS / "segment_profile.csv", index=False, float_format="%.4f")

    expected = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatch = hub_segments.loc[hub_segments.apply(
        lambda row: row["segment"] != expected[int(row["hub_tier"])], axis=1
    )].copy()
    mismatch.to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)
    print("Part D segmentation complete")
    print(profile.head(1).to_string(index=False))
    print(pd.DataFrame(separation_rows).to_string(index=False))
    print(segment_profile.to_string(index=False))
    print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
