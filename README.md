# Delhivery Shipment Delay — Capstone

## Project overview

This repository builds a reproducible shipment dataset and evidence base for the four-part Delhivery logistics capstone. Part A establishes the cleaned modelling table and delay evidence used by the later modelling and hub-segmentation parts.

## Environment setup

Use the exact package versions recorded in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Part A — Data Foundation and Delay Evidence

Run commands from the repository root, in this order:

```bash
python generate_data.py
python part_a_clean_eda.py
```

The generator is committed verbatim. It writes `data/hubs.csv`, `data/raw_shipments.csv`, and `data/new_shipments.csv`. The expected generator output is:

```text
24 1224 48
```

The Part A script creates `data/modelling_table.csv` and three evidence files under `outputs/`:

- `part_a_checks.csv`
- `part_a_delay_by_group.csv`
- `part_a_correlations.csv`

### Cleaning and feature construction

`src/shipment_prep.py` provides `clean_shipments(raw, medians=None)` and `build_features(clean, hubs)`. Cleaning removes exact duplicate rows, standardizes service names, fills missing payment mode with `Unknown`, parses declared value and dispatch time, treats parcel weights above 100 kg as scanner-fault sentinels, and imputes the three numeric columns with historical medians. The returned medians can be reused to prepare incoming shipments without recomputing imputation values.

The feature builder joins hub attributes and constructs the promised-hours, COD, weekend, dispatch-hour, and log-declared-value features. It does not depend on `delivery_hours`, so it can also be applied to new shipments.

### Data findings

The exact findings are recorded in the generated evidence files. The reference checks for this synthetic dataset are:

- The historical export contains 1,224 rows; exact duplicate removal removes 24 rows, leaving 1,200 unique shipments.
- There are 5 scanner-fault sentinel weights. The pre-fill missing counts are 36 for distance, 24 for parcel weight, 12 for declared value, and 18 for payment mode.
- The modelling table contains 1,200 rows after the hub merge and has zero missing values across the 17 model features, delivery-hours target, and delay label.
- There are 278 delayed shipments, for an overall delay rate of approximately 0.2317.

#### Delay rate by service type

| Service type | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| Economy | 315 | 27 | 0.0857 |
| Express | 243 | 119 | 0.4897 |
| Standard | 642 | 132 | 0.2056 |

Express has the highest delay rate at 0.4897.

#### Delay rate by hub tier

| Hub tier | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 1 | 400 | 16 | 0.0400 |
| 2 | 400 | 86 | 0.2150 |
| 3 | 400 | 176 | 0.4400 |

The delay rate rises from 0.0400 in Tier 1 to 0.4400 in Tier 3, an increase of 0.4000 (40.00 percentage points).

#### Delay rate by distance band

| Distance band (km) | Shipments | Delayed | Delay rate |
|---|---:|---:|---:|
| 0–200 | 274 | 6 | 0.0219 |
| 201–400 | 563 | 77 | 0.1368 |
| 401–600 | 275 | 120 | 0.4364 |
| 601+ | 88 | 75 | 0.8523 |

Delay rate increases across all four distance bands, from 0.0219 in the shortest band to 0.8523 in the longest band.

#### Correlation and declared-value distribution

The three strongest absolute correlations with `delivery_hours` are `promised_hours` (0.8372), `distance_km` (0.4013), and `n_stops` (0.3247). The reference spot-checks are `corr(delivery_hours, promised_hours) = 0.837`, `corr(delivery_hours, distance_km) = 0.401`, and `corr(is_delayed, distance_km) = 0.546`. The full numeric correlation matrix is in `outputs/part_a_correlations.csv`.

The median-filled `declared_value_inr` has mean ₹1,533.0542 and median ₹1,121.5000; its natural-log feature has mean 7.0277 and median 7.0224. The log transform compresses the long right tail of declared values, reducing the influence of unusually high values and making the scale less skewed.

## Reproducibility

Run the two commands above from a clean clone. Do not edit generated CSVs by hand. All CSV exports use `index=False` except the correlation matrix, which retains row labels as its first column. Numeric output values are written to four decimal places.

## Part B — Delivery-Hours Regression

Run from the repository root after completing Part A:

```bash
python part_b_regression.py
```

The script reads `data/modelling_table.csv` and uses the exact 17 features in the order defined by `src/shipment_prep.py`. It predicts `delivery_hours`. The split is 840 training rows, 180 validation rows, and 180 untouched test rows, using `random_state=42` for both split calls.

### Leakage controls

The diagnostic `StandardScaler` is fitted only on the 840 training rows. Its fitted parameters are then used to transform the test rows. The linear-regression candidate is also a scikit-learn pipeline, so its scaler is fitted inside the model pipeline on training data. The validation and test rows do not participate in fitting either scaler. The diagnostic output reports a maximum absolute scaled training-column mean of approximately 0 and a test-column mean magnitude of 0.2187, as expected when the scaler is not fitted on the test data.

### Model results (regression)

The mean baseline has validation R² **-0.0212**. This means predicting the training-target mean for every validation shipment performs slightly worse, in squared-error terms, than predicting the actual validation-target mean. It is a useful reference point, not a useful operational predictor.

**Linear regression wins validation**, with RMSE **4.8912 hours** and R² **0.9313**. Gradient boosting is very close (RMSE 4.9165 hours; R² 0.9306), followed by random forest (5.5095 hours; R² 0.9128) and the selected decision tree (6.2796 hours; R² 0.8867). The final model is therefore linear regression, as required by the rule to select the candidate with the lowest validation RMSE. Evaluated once on the untouched test set, it achieves RMSE **4.5235 hours** and R² **0.9493**.

The decision-tree depth diagnostic shows the bias/variance trade-off. Depth 1 is **high-bias**: train RMSE is 11.6334 hours and validation RMSE is 12.1324 hours, so both errors are high while their gap is only 0.4990 hours. Depth 6 is the validation winner (6.2796 hours). At depth 8, the train/validation gap widens to 3.4541 hours; depth 12 (gap 5.9927 hours) and the unlimited tree (gap 7.8368 hours) show **high variance/overfitting**. The unlimited tree memorises the training rows (train RMSE 0) but generalises less well.

The learning-rate experiment selects **0.1** (validation R² 0.9306). A rate of 0.01 is too slow for 100 boosting stages and underfits (R² 0.7127). A rate of 1.0 takes overly large boosting steps and overshoots, reducing validation R² to 0.8429.

The cross-validation ranking differs slightly from the validation ranking: linear regression has the lower validation RMSE, while gradient boosting has the higher 5-fold mean R² on the combined 1,020 training-plus-validation rows (0.9353 versus 0.9261). Both have low fold-to-fold variability (standard deviations 0.0045 and 0.0048). **Linear regression is the model to ship for this task** because the specification explicitly defines the final model as the candidate with the lowest validation RMSE; cross-validation is a robustness comparison, not permission to change that selection rule. The two models are close, so the CV result is useful evidence that gradient boosting is a credible alternative, but it does not override the prescribed selection procedure.

### Part B outputs

- `outputs/part_b_checks.csv` — split sizes and scaler leakage diagnostics.
- `outputs/part_b_tree_depth.csv` — training and validation RMSE by tree depth.
- `outputs/part_b_metrics.csv` — validation metrics for the baseline and four candidates, plus one test row for the selected final model.
- `outputs/part_b_gb_learning_rate.csv` — learning-rate comparison.
- `outputs/part_b_cv.csv` — shared 5-fold cross-validation comparison for the two best validation candidates.


## Part C — Delay Classifier and Alert Threshold

Run from the repository root after Part A:

```bash
python part_c_classification.py
```

The script repeats the two-step split with stratification on `is_delayed` in both calls. The split is 840 / 180 / 180 rows, containing 195 / 42 / 41 delayed parcels in train / validation / test. The blind baseline predicts every validation parcel on time: it reaches accuracy **0.7667** only because most parcels are on time, but its recall and F1 are both **0.0000**. It therefore identifies none of the parcels that operations needs to act on.

At the default 0.5 rule, logistic regression achieves validation accuracy **0.9056** and F1 **0.8046**; the selected depth-6 decision tree achieves accuracy **0.8833** and F1 **0.7470**; the default 17-feature random forest achieves accuracy **0.9056** and F1 **0.7848**. All trained models beat the blind baseline's accuracy, but recall and F1 are more informative than accuracy for the delayed minority class.

### Feature signal and selection

The training-only random forest assigns **0.2245** of total feature importance to the five bookkeeping/noise columns. All five rank 8–17 by univariate F-score, outside the top seven; their F-scores are all below 10. The top seven F-score features, in rank order, are `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag`. The `k` experiment supports **k = 7**: its validation F1 is **0.8333**, higher than k=5 (**0.6988**), k=10 (**0.7857**), k=12 (**0.8049**) and all 17 features (**0.7848**).

### Tuning and final model

On the same seven selected features, grid search evaluates 27 candidates / 135 fits and obtains CV accuracy **0.8952** with validation F1 **0.7750**. Randomized search evaluates 9 candidates / 45 fits and obtains CV accuracy **0.8905** with validation F1 **0.7848**. The untuned 100-tree forest on the same seven features has validation F1 **0.8333**, so neither tuning search beats it on the 180-row validation sample. This is a reminder that a more expensive search does not guarantee better held-out performance, particularly when the selection set is small. The prescribed validation-F1 rule therefore selects `default_forest_same_7_features` (100 trees, unlimited depth, minimum leaf size 1).

### Alert threshold and final test

The operations rule selects the **highest threshold with validation recall at least 0.80**, which is **0.5**. At this threshold, validation precision is **0.7955** and recall is **0.8333**: approximately 8 of every 10 alerts are truly delayed, and approximately 8 of every 10 delayed parcels are caught. Raising the threshold to 0.6 improves precision to 0.8889 but drops recall to 0.7619, below the operational requirement. The untouched test evaluation at threshold 0.5 gives precision **0.6889**, recall **0.7561**, and F1 **0.7209**. Thus the test set catches fewer delayed parcels than validation (0.7561 versus 0.8333); this difference should be interpreted cautiously because the test contains only 41 delayed parcels.

Part C evidence files: `part_c_checks.csv`, `part_c_models.csv`, `part_c_feature_scores.csv`, `part_c_selectk.csv`, `part_c_tuning.csv`, `part_c_thresholds.csv`, `part_c_final_config.csv`, and `part_c_final_test.csv` under `outputs/`.

## Part D — Hub Segments and Operational Actions

Run the segmentation, model packaging, and scoring scripts in this order:

```bash
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`part_d_segments.py` profiles the 24 hubs and clusters them using the four standardized hub-level operating measures. The fixed ascending `hub_id` order and `random_state=42`, `n_init=10` are intentional. The resulting segment labels are assigned only after ordering clusters by their mean delay rate: `Standard`, `Watch`, then `Escalate`.

### Operations recommendation

The `Escalate` segment has 9 hubs and a mean hub delay rate of **0.4600**. Its hubs are Guwahati (0.6200), Agartala (0.5400), Raipur (0.4800), Nagpur (0.4600), Patna (0.4600), Siliguri (0.4400), Bhubaneswar (0.4000), Ranchi (0.3800), and Kochi (0.3600). These hubs should receive the hub-manager playbook: daily review of delayed and high-probability parcels, capacity/lane review, and active exception ownership.

Four hubs do not behave like their nominal tier suggests. Kolkata (`HUB07`) is Tier 1 but is in `Watch`; Nagpur (`HUB13`) and Kochi (`HUB15`) are Tier 2 but are in `Escalate`; Dehradun (`HUB22`) is Tier 3 but is in `Watch`. Use the segment as the operational playbook assignment and the tier as a descriptive label, not as a substitute for observed hub performance. In particular, the two Tier 2 hubs in `Escalate` need the more intensive response, while the Tier 3 hub in `Watch` should not automatically receive the highest-intensity playbook.

The alert threshold remains **0.5**. On the final test set, precision **0.6889** means roughly **6.9 of every 10 flagged parcels** are actually delayed; recall **0.7561** means roughly **7.6 of every 10 delayed parcels** are caught. On the 48 new shipments, **8** are flagged; **6** of those require `Escalate to hub manager` and the other **2** receive `Proactive customer call`. Unflagged shipments receive `No action`.

The separation ratio increases from **2.8940** at K=2 to **5.3928** at K=3, and continues to rise at K=4 (**6.3047**) and K=5 (**6.8866**). K=3 is retained because operations has exactly three playbooks and the K=4 and K=5 solutions contain clusters smaller than 6 hubs; those small groups are not large enough to justify a separate operational playbook. The pinned order and seed also reproduce the reference cluster sizes: 15-9, 9-8-7, 8-7-5-4, and 8-7-4-3-2 for K=2 through K=5.

### Model packaging and scoring

`package_model.py` reads `outputs/part_c_final_config.csv`, rebuilds the same stratified split, fits the final seven-feature pipeline on the training rows, and writes `models/delay_model.pkl`, `models/delay_threshold.csv`, `models/feature_order.csv`, and `models/impute_values.csv`. It reloads the pickle and checks its test probabilities against the in-memory pipeline: **180** test rows are scored and the maximum absolute probability difference is **0.0000000000**. The model pickle is regenerated by running `python package_model.py`; a committed pickle loads reliably only with the scikit-learn version pinned in `requirements.txt`. If the installed version differs, the grader should rerun `package_model.py` rather than load the committed pickle.

`predict_delay.py` uses `clean_shipments` and `build_features` from `src/shipment_prep.py`, reuses the stored historical medians and feature order, applies the stored threshold, and joins the hub segment. It writes `outputs/new_shipment_actions.csv` in the original new-shipment order. The script is deterministic and can be run repeatedly to regenerate the same action file.

## How to run

From a fresh clone, run the following commands from the repository root, in order:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

The generator recreates the historical, hub, and new-shipment inputs. Each later script writes its outputs from those inputs; do not edit generated CSVs by hand. `models/delay_model.pkl` is regenerated by running `python package_model.py`. The committed pickle is version-dependent and should be loaded with the scikit-learn version pinned in `requirements.txt`; if the installed version differs, regenerate it with `package_model.py` before scoring.

## Production plan

- **Accuracy:** the repository reports regression test RMSE/R² and classification test precision/recall/F1 in `outputs/part_b_metrics.csv` and `outputs/part_c_final_test.csv`; these are measurable acceptance baselines, not guarantees of future performance.
- **Reliability:** split-count assertions, feature/target validation in the preparation flow, the package reload probability check, and scoring assertions for missing/duplicate IDs catch common failure modes.
- **Reproducibility:** random seeds are fixed in the split, models, searches and K-Means; package versions are pinned in `requirements.txt`; the generator and scripts recreate the outputs.
- **Maintainability:** preparation logic is centralized in `src/shipment_prep.py`, while regression, classification, segmentation, packaging and prediction are separate scripts.
- **Consistency:** training and inference share `clean_shipments` / `build_features`; inference reads the persisted historical medians and ordered feature list instead of fitting new imputations or reconstructing the feature order independently.
- **Monitorability:** retain weekly shipment outcomes, model probabilities, flags, hub segments and actions so delayed share, flag volume, precision and recall can be calculated once outcomes mature.
- **Performance:** prediction is a batch pipeline using vectorized pandas transformations and a compact seven-feature forest. Benchmark end-to-end runtime and throughput on production-sized batches before setting an SLA; the capstone data is too small to establish a production performance target.

Suggested weekly operating triggers (review and investigate rather than automatically retrain):

1. **Delivered-shipment delay share:** investigate if the rolling weekly delayed share exceeds **0.30** for two consecutive weeks, compared with the historical **0.2317** overall reference in Part A.
2. **Flagged share:** investigate if more than **0.25** of weekly shipments are flagged at the stored threshold; check for changed mix, weather, service mix or feature drift.
3. **Matured-outcome recall:** investigate and consider threshold/model review if rolling recall falls below **0.75** once enough flagged and unflagged shipments have completed delivery and labels are reliable.
4. **Feature drift:** compare weekly feature distributions with the training reference; investigate a population stability index above **0.20** for a material feature, or a sustained shift in the share of long-distance, rainy, sale-event or hub-tier shipments.

Retrain on a quarterly cadence, with an earlier review after a material process or data change. For the **15–28 September sale window**, evaluate performance against the same sale-event period (or the closest comparable sale event), not ordinary weeks: `sale_event_flag` captures a distinct operating regime and sale traffic changes the shipment mix and delay mechanism. Data drift would appear as sustained changes in distance, stops, service mix, rain/sale incidence, hub capacity or the relationship between those inputs and observed delay. Recalibrate only after enough sale-event outcomes mature; keep the event flag and monitor sale and non-sale cohorts separately.

## Limitations

- The dataset is synthetic and reflects the generator's assumptions; real logistics networks may contain additional causes of delay, missingness patterns and operational constraints.
- The test set contains only 41 delayed parcels, so a small number of changed predictions can materially move recall, precision and F1.
- The threshold and final configuration were selected using one 180-row validation split. They may not generalize to a different season, region or operating regime; repeated temporal and event-based validation is needed before production deployment.
- Hub segments are descriptive clusters of 24 hubs using four aggregated measures. Cluster membership is sensitive to the chosen variables and should be reviewed when hub operations materially change.
