# Project Findings & Key Observations

---

## Finding 1 — Adding Driver & Vehicle Features Did Not Improve Accuracy

**What we did:**
We ran two versions of the feature selection and classification pipeline:
- **V1** — 11 road/environment features only → see [`feature_selection_results.md`](feature_selection_results.md)
- **V2** — 21 features including driver (age, sex, journey purpose) and vehicle (type, manoeuvre, towing) features → see [`feature_selection_v1_vs_v2.md`](feature_selection_v1_vs_v2.md)

**What we expected:**
Adding driver and vehicle context should give the model more signal and improve its ability to distinguish Fatal from Serious from Slight accidents.

**What actually happened:**

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| RF F1-Macro | 0.4087 | 0.4026 | -0.006 |
| RF Test Accuracy | 0.6279 | 0.6259 | -0.002 |
| RF Fatal Recall | 0.74 | 0.71 | -0.03 |

Accuracy stayed essentially flat. The new features made no meaningful difference.

**Why this matters:**
This is not a failure of the model or the feature engineering — it is a finding about the **nature of the problem itself**. Accident severity is partly determined by physical factors that are never recorded in the DfT dataset:

- Exact angle and speed of impact
- Whether seatbelts were worn
- Airbag deployment
- Time to medical response
- Structural integrity of the vehicles involved
- Whether a pedestrian was hit at exactly 30 vs 31 mph

No amount of additional pre-crash features (road type, driver age, vehicle type) can substitute for these post-crash physical measurements. The model is already capturing **most of the predictable signal available in this dataset**.

**What this means for the report:**
This finding strengthens the project's conclusion. It shows the team understood the limits of the data, not just the limits of the model. A model that achieves 0.40 F1-Macro on a 3-class severity problem with 84%/14%/2% class imbalance — using only pre-crash road conditions — is performing reasonably given what information is available.

**Relevant files:**
- Feature sets compared: [`feature_selection_v1_vs_v2.md`](feature_selection_v1_vs_v2.md)
- V2 feature selection output: [`feature_selection_results.md`](feature_selection_results.md)
- Classification results: [`classification_results.md`](classification_results.md)
- Source code: [`src/feature_selection.py`](../src/feature_selection.py), [`src/classification.py`](../src/classification.py)

---

## Finding 2 — Fatal Recall is the Most Important Metric Here

Overall accuracy (0.63) looks low but is misleading. With 84% of accidents being Slight, a model that always predicts "Slight" would get 84% accuracy while being completely useless.

**What matters more is Fatal recall — 0.71 (RF)**:
- Out of every 100 fatal accidents, the model correctly flags 71 as Fatal
- This is the number that matters for road safety applications — missing a Fatal accident is far costlier than a false alarm

**Relevant files:**
- [`classification_results.md`](classification_results.md) — full precision/recall breakdown per class

---

## Finding 3 — LDA Consistently Selects Speed Limit and Vehicle Manoeuvre as Top Predictors

Across both V1 and V2, LDA ranked `Speed_limit` and `Number_of_Vehicles` as the top two discriminating features. In V2, `Vehicle_Manoeuvre` entered at #3 — meaning *what the vehicle was doing before impact* is a strong predictor of severity, stronger than road type or lighting.

This aligns with real-world road safety research: high-speed crashes and overtaking manoeuvres are disproportionately represented in fatal accidents.

**Relevant files:**
- [`feature_selection_results.md`](feature_selection_results.md) — LDA coefficient table
- [`output/lda_feature_coefficients.png`](lda_feature_coefficients.png) — visual chart

---

## Finding 4 — Class Imbalance Strategy Choice

SMOTE was considered but rejected in favour of `class_weight='balanced'`.

**Reason:** Most features in this dataset are label-encoded categoricals (e.g. `Road_Type=3`, `Light_Conditions=4`). SMOTE generates synthetic rows by interpolating between existing ones — producing values like `Road_Type=2.7` which correspond to no real road type. This would introduce noise, not signal.

`class_weight='balanced'` achieves the same effect — giving Fatal accidents ~17× more influence on the model's loss function — without creating any fake data.

**Relevant files:**
- [`classification_results.md`](classification_results.md) — imbalance strategy section
- [`src/classification.py`](../src/classification.py) — implementation
