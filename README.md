# UK Traffic Accident Analysis — Big Data Final Project

Analyzes UK DfT road accident data (2005–2015) to identify high-risk patterns,
predict accident severity, discover geographic hotspots, and forecast casualty counts.

---

## Project Structure

```
ai-accident-insights-UK/
├── data/                   # Raw input CSVs (download from Kaggle — never modified)
│   └── contextCSVs/        # Lookup tables for coded column values
├── output/                 # All generated files (CSVs, plots, result tables)
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── classification.py
│   ├── clustering.py
│   ├── regression.py
│   └── visualizations.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 1. Download the Dataset

1. Go to the Kaggle dataset page:
   **https://www.kaggle.com/datasets/silicon99/dft-accident-data**

2. Click **Download** (you'll need a free Kaggle account).

3. Unzip the downloaded file:
   ```bash
   unzip dft-accident-data.zip -d data/
   ```

4. Make sure these three files are inside the `data/` folder:
   ```
   data/
   ├── Accidents0515.csv
   ├── Casualties0515.csv
   └── Vehicles0515.csv
   ```

> The `data/` folder is for **raw input only**. Never save generated files here.

---

## 2. Create a Virtual Environment

Using **conda** (recommended):

```bash
conda create -n ai_accident-insights_uk python=3.11 -y
conda activate ai_accident-insights_uk
```

Or using **venv**:

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Packages installed:** pandas, numpy, scikit-learn, xgboost, imbalanced-learn, matplotlib, seaborn, scipy, joblib

---

## 4. Run the Pipeline

Run each phase individually and verify before moving to the next:

```bash
# Phase 1 — Clean and merge the raw data
python src/preprocessing.py

# Phase 2 — Select features, run PCA
python src/feature_selection.py

# Phase 3 — Train classifiers (predict accident severity)
python src/classification.py

# Phase 4 — Cluster accidents, find geographic hotspots
python src/clustering.py

# Phase 5 — Regression (predict number of casualties)
python src/regression.py

# Phase 6 — Final visualizations and summary
python src/visualizations.py
```

Or run the full pipeline end-to-end:

```bash
python main.py
```

---

## 5. Output

All generated files are saved to `output/`:

| File | Created By | Description |
|------|-----------|-------------|
| `accidents_cleaned.csv` | preprocessing.py | Merged, cleaned master dataset |
| `X_final.csv` | feature_selection.py | Final feature matrix (top 10 features) |
| `y_final.csv` | feature_selection.py | Target variable (Accident_Severity) |
| `X_pca.csv` | feature_selection.py | 2D PCA result for clustering |
| `lat_lon.csv` | feature_selection.py | Latitude/Longitude for geographic map |
| `correlation_matrix.png` | feature_selection.py | Feature correlation heatmap |
| `feature_importance.png` | feature_selection.py | Top 10 features bar chart |
| `confusion_matrix.png` | classification.py | Best classifier confusion matrix |
| `roc_curves.png` | classification.py | ROC curves for 3 severity classes |
| `classification_results.txt` | classification.py | Model comparison table |
| `elbow_plot.png` | clustering.py | K-Means elbow method |
| `geographic_hotspots.png` | clustering.py | UK map colored by cluster |
| `clustering_results.txt` | clustering.py | Silhouette + Davies-Bouldin scores |
| `regression_actual_vs_predicted.png` | regression.py | Actual vs predicted scatter |
| `regression_results.txt` | regression.py | RMSE, MAE, R² comparison |
| `data_distributions.png` | visualizations.py | 6-panel distribution dashboard |
| `preprocessing_result.md` | preprocessing.py | Preprocessing summary stats |

---

## Dataset Info

| File | Rows | Columns |
|------|------|---------|
| `Accidents0515.csv` | 1,780,653 | 32 |
| `Casualties0515.csv` | 2,402,909 | 15 |
| `Vehicles0515.csv` | 3,262,270 | 22 |
| **Merged (cleaned)** | **4,427,649** | **72** |

**Accident Severity coding (DfT):** 1 = Fatal, 2 = Serious, 3 = Slight

**Date range:** 2005-01-01 → 2015-12-31
