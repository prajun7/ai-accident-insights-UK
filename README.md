# UK Traffic Accident Analysis — Big Data Final Project

Analyzes UK DfT road accident data (2005–2014) to identify high-risk patterns,
predict accident severity, discover geographic hotspots, and forecast casualty counts.

## Setup

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/silicon99/dft-accident-data
2. Place `Accidents0514.csv`, `Casualties0514.csv`, `Vehicles0514.csv` in the `data/` folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run

```bash
# Full pipeline
python main.py

# Individual phases
python src/preprocessing.py
python src/feature_selection.py
python src/classification.py
python src/clustering.py
python src/regression.py
python src/visualizations.py
```

## Output

All generated files (CSVs, plots, result tables) are saved to `outputs/`.

## Project Structure

```
uk_accident_project/
├── data/               # Input CSVs (download from Kaggle)
├── outputs/            # Generated outputs
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── classification.py
│   ├── clustering.py
│   ├── regression.py
│   └── visualizations.py
├── main.py
└── requirements.txt
```
