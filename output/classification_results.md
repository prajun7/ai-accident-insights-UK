# Classification Results — V2 (22 input columns → 10 LDA features)

## Input
- **X_final.csv:** 4,427,649 rows × 10 features
- **y_final.csv:** 4,427,649 labels (Accident_Severity)
- **Features used:** Speed_limit, Number_of_Vehicles, Vehicle_Manoeuvre, Road_Type, IsNight, Urban_or_Rural_Area, Sex_of_Driver, Junction_Detail, Age_Band_of_Driver, Light_Conditions

---

## Class Distribution
| Class | Label | Count | % |
|-------|-------|-------|---|
| 1 | Fatal | 85,223 | 1.9% |
| 2 | Serious | 616,609 | 13.9% |
| 3 | Slight | 3,725,817 | 84.1% |

---

## Train / Test Split
| Set | Rows | Columns |
|-----|------|---------|
| X_train | 3,542,119 | 10 |
| X_test | 885,530 | 10 |

- Split: 80% train / 20% test, stratified by class

---

## Class Imbalance Strategy
- **Method:** `class_weight='balanced'` — no synthetic data created
- **Why not SMOTE:** Features are label-encoded categoricals. SMOTE interpolates between rows, producing values like `Road_Type=1.7` which have no real meaning
- **Sample weight range:** min=0.3961 → max=17.3180
  - Fatal accidents receive ~17× more penalty weight than Slight during training

---

## Model Results

### Logistic Regression (Baseline)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.05 | 0.68 | 0.10 | 17,045 |
| Serious | 0.15 | 0.21 | 0.18 | 123,322 |
| Slight | 0.89 | 0.59 | 0.71 | 745,163 |
| **Macro avg** | **0.37** | **0.49** | **0.33** | 885,530 |
| Weighted avg | 0.77 | 0.54 | 0.63 | 885,530 |

- Train Accuracy: 0.5403 | Test Accuracy: 0.5408 | F1-Macro: **0.3280**

---

### Random Forest (200 trees, class_weight='balanced') — Best Model
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.08 | 0.71 | 0.14 | 17,045 |
| Serious | 0.24 | 0.37 | 0.29 | 123,322 |
| Slight | 0.91 | 0.67 | 0.77 | 745,163 |
| **Macro avg** | **0.41** | **0.58** | **0.40** | 885,530 |
| Weighted avg | 0.80 | 0.63 | 0.69 | 885,530 |

- Train Accuracy: 0.6397 | Test Accuracy: 0.6259 | F1-Macro: **0.4026**

**Confusion Matrix:**
|  | Pred Fatal | Pred Serious | Pred Slight |
|--|-----------|-------------|------------|
| **Actual Fatal** | 12,185 | 2,694 | 2,166 |
| **Actual Serious** | 32,492 | 46,098 | 44,732 |
| **Actual Slight** | 106,803 | 142,376 | 495,984 |

---

### XGBoost (200 trees, sample_weight='balanced')
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fatal | 0.07 | 0.72 | 0.12 | 17,045 |
| Serious | 0.21 | 0.31 | 0.25 | 123,322 |
| Slight | 0.91 | 0.64 | 0.75 | 745,163 |
| **Macro avg** | **0.40** | **0.55** | **0.37** | 885,530 |
| Weighted avg | 0.80 | 0.59 | 0.67 | 885,530 |

- Train Accuracy: 0.5944 | Test Accuracy: 0.5942 | F1-Macro: **0.3743**

**Confusion Matrix:**
|  | Pred Fatal | Pred Serious | Pred Slight |
|--|-----------|-------------|------------|
| **Actual Fatal** | 12,220 | 2,722 | 2,103 |
| **Actual Serious** | 40,800 | 37,934 | 44,588 |
| **Actual Slight** | 126,920 | 142,204 | 476,039 |

---

## Model Comparison Summary
| Model | Train Acc | Test Acc | F1-Macro |
|-------|-----------|----------|----------|
| Logistic Regression | 0.5403 | 0.5408 | 0.3280 |
| XGBoost | 0.5944 | 0.5942 | 0.3743 |
| **Random Forest** | **0.6397** | **0.6259** | **0.4026** |

**Best model: Random Forest** (highest F1-Macro: 0.4026)

---

## V1 vs V2 Classification Comparison
| Metric | V1 (11 features) | V2 (21 features → top 10) | Change |
|--------|-----------------|--------------------------|--------|
| LR F1-Macro | 0.3190 | 0.3280 | +0.009 |
| RF F1-Macro | 0.4087 | 0.4026 | -0.006 |
| XGB F1-Macro | 0.3752 | 0.3743 | -0.001 |
| RF Test Accuracy | 0.6279 | 0.6259 | -0.002 |
| RF Fatal Recall | 0.74 | 0.71 | -0.03 |
| RF Serious Recall | 0.39 | 0.37 | -0.02 |

> V2 added driver and vehicle features but accuracy stayed similar. This suggests the inherent difficulty of the problem — accident severity is hard to predict from pre-crash conditions alone, regardless of how many features are added. The LDA feature selection likely captured the most discriminating signal in both versions.

---

## Random Forest: Accuracy vs Number of Estimators
| n_estimators | Train Acc | Test Acc |
|-------------|-----------|----------|
| 10 | 0.6902 | 0.6374 |
| 30 | 0.6955 | 0.6427 |
| 50 | 0.6974 | 0.6443 |
| 100 | 0.6965 | 0.6434 |
| 150 | 0.6968 | 0.6435 |
| 200 | 0.6976 | 0.6442 |

> Performance plateaus after ~30–40 estimators. 200 used for the final model.

---

## Output Files
| File | Description |
|------|-------------|
| `confusion_matrix.png` | Best model (RF) confusion matrix heatmap |
| `roc_curves.png` | One-vs-rest ROC curves for all 3 severity classes |
| `accuracy_vs_estimators.png` | RF train/test accuracy vs n_estimators |
| `classification_results.txt` | Raw comparison table |
