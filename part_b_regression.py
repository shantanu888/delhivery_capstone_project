from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "delivery_hours"


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def write_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.4f")


def main():
    table = pd.read_csv(DATA / "modelling_table.csv")
    missing = [column for column in FEATURES + [TARGET] if column not in table.columns]
    if missing:
        raise ValueError(f"modelling_table.csv is missing required columns: {missing}")

    X = table[FEATURES].copy()
    y = table[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    if (len(X_train), len(X_val), len(X_test)) != (840, 180, 180):
        raise AssertionError("Unexpected split sizes; expected 840 / 180 / 180.")

    # Diagnostic scaler: fit only on training rows, then transform validation/test.
    diagnostic_scaler = StandardScaler()
    X_train_scaled = diagnostic_scaler.fit_transform(X_train)
    X_test_scaled = diagnostic_scaler.transform(X_test)
    checks = [
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
        ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max())),
    ]
    write_csv(pd.DataFrame(checks, columns=["metric", "value"]), OUTPUTS / "part_b_checks.csv")

    # Mean baseline, scored on validation only.
    baseline_prediction = np.full(len(y_val), y_train.mean())
    metric_rows = [{
        "model": "mean_baseline",
        "split": "val",
        **regression_metrics(y_val, baseline_prediction),
    }]

    # Tree-depth diagnostic. The unlimited tree is evaluated last.
    depth_options = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    depth_models = {}
    for depth in depth_options:
        tree = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        train_prediction = tree.predict(X_train)
        val_prediction = tree.predict(X_val)
        train_rmse = float(np.sqrt(mean_squared_error(y_train, train_prediction)))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_prediction)))
        depth_rows.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
        })
        depth_models[depth] = tree

    depth_frame = pd.DataFrame(depth_rows)
    write_csv(depth_frame, OUTPUTS / "part_b_tree_depth.csv")
    # Stable sort preserves the requested smaller-depth tie-break.
    best_depth_row = min(
        depth_rows,
        key=lambda row: (row["val_rmse"], float("inf") if row["max_depth"] == "unlimited" else row["max_depth"])
    )
    best_depth = None if best_depth_row["max_depth"] == "unlimited" else int(best_depth_row["max_depth"])
    best_tree = depth_models[best_depth]

    candidates = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "decision_tree": best_tree,
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }

    validation_results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        metrics = regression_metrics(y_val, prediction)
        validation_results[name] = metrics
        metric_rows.append({"model": name, "split": "val", **metrics})

    metrics_frame = pd.DataFrame(
        metric_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
    )

    # Learning-rate experiment; each model is independently trained on training rows.
    learning_rows = []
    for learning_rate in [0.01, 0.1, 1.0]:
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=learning_rate, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        metrics = regression_metrics(y_val, prediction)
        learning_rows.append({
            "learning_rate": learning_rate,
            "val_rmse": metrics["RMSE"],
            "val_r2": metrics["R2"],
        })
    write_csv(
        pd.DataFrame(learning_rows, columns=["learning_rate", "val_rmse", "val_r2"]),
        OUTPUTS / "part_b_gb_learning_rate.csv",
    )

    # Compare the two best validation candidates on the same 1,020-row set
    # and with the exact same shuffled KFold splitter.
    ranked_names = sorted(
        validation_results,
        key=lambda name: validation_results[name]["RMSE"]
    )
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    shared_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name in ranked_names[:2]:
        cv_scores = cross_val_score(
            candidates[name], X_train_val, y_train_val, cv=shared_kfold
        )
        cv_rows.append({
            "model": name,
            "mean_r2": float(np.mean(cv_scores)),
            "std_r2": float(np.std(cv_scores)),
        })
    write_csv(
        pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"]),
        OUTPUTS / "part_b_cv.csv",
    )

    # Only the validation winner is evaluated on the untouched test set, once.
    final_name = ranked_names[0]
    final_model = candidates[final_name]  # fitted on training rows above
    test_prediction = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_prediction)
    metrics_frame = pd.concat([
        metrics_frame,
        pd.DataFrame([{"model": final_name, "split": "test", **test_metrics}]),
    ], ignore_index=True)
    write_csv(metrics_frame, OUTPUTS / "part_b_metrics.csv")

    # Acceptance checks derived from actual outputs.
    assert abs(float(X_train_scaled.mean(axis=0).max())) < 1e-9
    assert float(np.abs(X_train_scaled.mean(axis=0)).max()) < 1e-9
    assert float(np.abs(X_test_scaled.mean(axis=0)).max()) > 0.01
    assert len(depth_frame) == 8
    train_errors = depth_frame["train_rmse"].to_numpy()
    assert np.all(np.diff(train_errors) <= 1e-8)
    assert abs(
        validation_results["decision_tree"]["RMSE"] - depth_frame["val_rmse"].min()
    ) <= 0.001
    assert len(metrics_frame) == 6
    assert len(cv_rows) == 2
    assert len(learning_rows) == 3
    assert np.allclose(
        metrics_frame["RMSE"].to_numpy(),
        np.sqrt(metrics_frame["MSE"].to_numpy()),
        atol=0.001,
    )

    print(f"Part B complete. Validation winner: {final_name}")
    print(f"Best tree depth: {'unlimited' if best_depth is None else best_depth}")
    print("\nChecks:")
    print(pd.read_csv(OUTPUTS / "part_b_checks.csv").to_string(index=False))
    print("\nTree depths:")
    print(pd.read_csv(OUTPUTS / "part_b_tree_depth.csv").to_string(index=False))
    print("\nMetrics:")
    print(pd.read_csv(OUTPUTS / "part_b_metrics.csv").to_string(index=False))
    print("\nLearning rates:")
    print(pd.read_csv(OUTPUTS / "part_b_gb_learning_rate.csv").to_string(index=False))
    print("\nCross-validation:")
    print(pd.read_csv(OUTPUTS / "part_b_cv.csv").to_string(index=False))


if __name__ == "__main__":
    main()
