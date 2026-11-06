from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def stratified_split(frame):
    X = frame[FEATURES]
    y = frame["is_delayed"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def metric_row(name, y_true, y_pred, precision_override=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": (precision_score(y_true, y_pred, zero_division=0)
                      if precision_override is None else precision_override),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def threshold_row(threshold, y_true, probabilities):
    pred = (probabilities >= threshold).astype(int)
    row = metric_row(str(threshold), y_true, pred)
    row["threshold"] = threshold
    return {k: row[k] for k in ["threshold", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]}


def main():
    frame = pd.read_csv(DATA / "modelling_table.csv")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(frame)
    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
    assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)

    # Task 2: validation-only default-threshold model comparison.
    baseline = np.zeros(len(y_val), dtype=int)
    rows = [metric_row("blind_all_on_time", y_val, baseline, precision_override=0)]

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        logistic.fit(X_train, y_train)
    rows.append(metric_row("logistic_regression", y_val, logistic.predict(X_val)))

    tree_scores = []
    for depth in [2, 3, 4, 5, 6, 8, 10]:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        pred = tree.predict(X_val)
        tree_scores.append((f1_score(y_val, pred, zero_division=0), depth, tree, pred))
    best_tree_f1 = max(x[0] for x in tree_scores)
    best_tree = min((x for x in tree_scores if np.isclose(x[0], best_tree_f1)), key=lambda x: x[1])
    rows.append(metric_row(f"decision_tree_depth_{best_tree[1]}", y_val, best_tree[3]))

    default_rf_all = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    default_rf_all.fit(X_train, y_train)
    rows.append(metric_row("random_forest_default", y_val, default_rf_all.predict(X_val)))
    pd.DataFrame(rows).to_csv(OUTPUTS / "part_c_models.csv", index=False, float_format="%.4f")

    # Task 3: feature importance and univariate F scores, fit on training rows only.
    importance_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    importance_model.fit(X_train, y_train)
    selector_all = SelectKBest(score_func=f_classif, k="all").fit(X_train, y_train)
    f_scores = np.nan_to_num(selector_all.scores_, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-f_scores, kind="stable")
    ranks = np.empty(len(FEATURES), dtype=int)
    ranks[order] = np.arange(1, len(FEATURES) + 1)
    feature_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": f_scores,
        "f_rank": ranks,
        "rf_importance": importance_model.feature_importances_,
        "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
    })
    feature_scores.to_csv(OUTPUTS / "part_c_feature_scores.csv", index=False, float_format="%.4f")

    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val), zero_division=0)})
    pd.DataFrame(selectk_rows).to_csv(OUTPUTS / "part_c_selectk.csv", index=False, float_format="%.4f")

    # Task 4: selector fit on training rows only; validation rows are only transformed.
    selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
    X_train_7 = selector.transform(X_train)
    X_val_7 = selector.transform(X_val)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE)
    grid = GridSearchCV(base_rf, param_grid=param_grid, cv=5)
    grid.fit(X_train_7, y_train)
    randomized = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_grid, n_iter=9, cv=5, random_state=RANDOM_STATE,
    )
    randomized.fit(X_train_7, y_train)

    tuned_default = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                           max_depth=None, min_samples_leaf=1)
    tuned_default.fit(X_train_7, y_train)
    tuning_rows = [
        {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 135,
         "best_params": json.dumps(grid.best_params_, sort_keys=True),
         "best_cv_accuracy": grid.best_score_,
         "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_7), zero_division=0)},
        {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 45,
         "best_params": json.dumps(randomized.best_params_, sort_keys=True),
         "best_cv_accuracy": randomized.best_score_,
         "val_f1": f1_score(y_val, randomized.best_estimator_.predict(X_val_7), zero_division=0)},
        {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
         "best_params": "defaults", "best_cv_accuracy": np.nan,
         "val_f1": f1_score(y_val, tuned_default.predict(X_val_7), zero_division=0)},
    ]
    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(OUTPUTS / "part_c_tuning.csv", index=False, float_format="%.4f")

    # Task 5: prescribed tie priority: default, grid, randomized.
    priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
    winner = sorted(tuning_rows, key=lambda r: (-r["val_f1"], priority[r["search"]]))[0]
    if winner["search"] == "grid_search":
        chosen_params = grid.best_params_
    elif winner["search"] == "randomized_search":
        chosen_params = randomized.best_params_
    else:
        chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE, **chosen_params)),
    ])
    final_model.fit(X_train, y_train)

    # Task 6: exact decimal thresholds and recall-constrained highest threshold.
    val_prob = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_df = pd.DataFrame([threshold_row(t, y_val, val_prob) for t in thresholds])
    threshold_df.to_csv(OUTPUTS / "part_c_thresholds.csv", index=False, float_format="%.4f")
    eligible = threshold_df.loc[threshold_df["recall"] >= 0.80, "threshold"]
    chosen_threshold = float(eligible.max()) if len(eligible) else 0.1

    # Task 7: touch the test split exactly once, at the selected threshold.
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= chosen_threshold).astype(int)
    final_test = metric_row("final_test", y_test, test_pred)
    final_test_row = {"threshold": chosen_threshold, **{k: final_test[k] for k in
        ["TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]}}
    pd.DataFrame([final_test_row]).to_csv(OUTPUTS / "part_c_final_test.csv", index=False, float_format="%.4f")

    depth_value = "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"]
    pd.DataFrame([{
        "selector_k": 7, "n_estimators": chosen_params["n_estimators"],
        "max_depth": depth_value, "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold, "chosen_from": winner["search"],
    }]).to_csv(OUTPUTS / "part_c_final_config.csv", index=False)

    checks = [
        ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
        ("train_delayed", int(y_train.sum())), ("val_delayed", int(y_val.sum())),
        ("test_delayed", int(y_test.sum())), ("chosen_tree_depth", best_tree[1]),
        ("chosen_tuning_candidate", winner["search"]), ("chosen_threshold", chosen_threshold),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(OUTPUTS / "part_c_checks.csv", index=False)
    print("Part C complete")
    print("Split sizes/delayed:", len(X_train), len(X_val), len(X_test), int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))
    print("Tuning:", tuning_df[["search", "val_f1", "best_cv_accuracy"]].to_string(index=False))
    print("Chosen:", winner["search"], chosen_params, "threshold", chosen_threshold)
    print("Final test:", final_test_row)


if __name__ == "__main__":
    main()
