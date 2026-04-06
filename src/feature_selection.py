import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load ---")
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'accidents_cleaned.csv'), low_memory=False)
    print(f"Loaded shape: {df.shape}")

    # ------------------------------------------------------------------
    # STEP 2 — SELECT COLUMNS
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Select Columns ---")

    # Save lat/lon before dropping
    lat_lon = df[['Latitude', 'Longitude']].copy()
    lat_lon.to_csv(os.path.join(OUTPUT_DIR, 'lat_lon.csv'), index=False)
    print(f"Saved lat_lon.csv — shape: {lat_lon.shape}")

    # Month, DayOfWeek, IsWeekend are dropped intentionally —
    # they capture accident *frequency* patterns (when accidents happen)
    # but not accident *severity* patterns (how bad they are).
    # Keeping them causes temporal features to dominate feature importance
    # and shifts model focus away from road condition predictors.
    # Hour and IsNight are kept because visibility directly affects severity.
    keep_cols = [
        'Accident_Severity',
        'Speed_limit', 'Road_Type', 'Light_Conditions',
        'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
        'Junction_Detail', 'Junction_Control', 'Number_of_Vehicles',
        'Hour', 'IsNight'
        # Number_of_Casualties intentionally excluded — data leakage:
        # casualty count is only known after the accident, same time as severity.
        # Month, DayOfWeek, IsWeekend excluded — frequency features, not severity features.
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    missing = [c for c in [
        'Accident_Severity', 'Speed_limit', 'Road_Type', 'Light_Conditions',
        'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
        'Junction_Detail', 'Junction_Control', 'Number_of_Vehicles',
        'Hour', 'IsNight'
    ] if c not in df.columns]
    if missing:
        print(f"Warning — columns not found (skipped): {missing}")

    df = df[keep_cols].copy()
    print(f"Shape after column selection: {df.shape}")
    print(f"Columns kept: {keep_cols}")
    print("\nNote: Month, DayOfWeek, IsWeekend removed (frequency not severity predictors).")
    print("Note: Number_of_Casualties removed (data leakage — known only after accident).")

    # ------------------------------------------------------------------
    # STEP 3 — LABEL ENCODING
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Label Encoding ---")
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            print(f"  Encoding '{col}': {df[col].nunique()} unique values → integers")
            df[col] = le.fit_transform(df[col].astype(str))

    # ------------------------------------------------------------------
    # STEP 4 — CORRELATION MATRIX
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Correlation Matrix ---")
    corr = df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
    plt.close()
    print("Saved correlation_matrix.png")

    # Drop one column from highly correlated pairs (|corr| > 0.85)
    upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [col for col in upper.columns if any(upper[col] > 0.85)]
    if drop_corr:
        print(f"Dropping highly correlated columns (|corr| > 0.85): {drop_corr}")
        df.drop(columns=drop_corr, inplace=True)
    else:
        print("No columns dropped for high correlation.")

    # ------------------------------------------------------------------
    # STEP 5 — SEPARATE TARGET
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Separate Target ---")
    X = df.drop(columns=['Accident_Severity'])
    y = df['Accident_Severity']
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index().rename({1:'Fatal',2:'Serious',3:'Slight'})}")

    # ------------------------------------------------------------------
    # STEP 6 — LDA FOR DIMENSIONALITY REDUCTION & FEATURE RANKING
    # ------------------------------------------------------------------
    # LDA (Linear Discriminant Analysis) is chosen over PCA here because:
    # - We have labeled classes (Slight / Serious / Fatal) — LDA uses them
    # - LDA finds directions that MAXIMIZE separation between severity classes
    # - PCA ignores class labels and maximizes general variance instead
    # - With 4.4M rows and 3 classes, LDA is highly stable
    # Max LDA components = n_classes - 1 = 2 (one per boundary between 3 classes)
    # ------------------------------------------------------------------
    print("\n--- STEP 6: LDA — Dimensionality Reduction ---")

    scaler_lda = StandardScaler()
    X_scaled_lda = scaler_lda.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled_lda, y)

    print(f"LDA explained variance ratio: LD1={lda.explained_variance_ratio_[0]:.3f}, "
          f"LD2={lda.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance explained by LDA: {lda.explained_variance_ratio_.sum():.3f}")

    # --- Feature coefficients (which original features drive each axis) ---
    scalings_df = pd.DataFrame(
        lda.scalings_,
        index=X.columns,
        columns=['LD1', 'LD2']
    )
    scalings_df['LD1_abs'] = scalings_df['LD1'].abs()
    scalings_df['LD2_abs'] = scalings_df['LD2'].abs()
    scalings_df_sorted = scalings_df.sort_values('LD1_abs', ascending=False)

    print(f"\nLDA Feature Coefficients (sorted by |LD1|):\n{scalings_df_sorted[['LD1','LD2']].round(4)}")

    # --- Plot: LDA LD1 feature coefficients (replaces RF importance bar chart) ---
    plt.figure(figsize=(10, 6))
    scalings_df_sorted['LD1_abs'].sort_values().plot(kind='barh', color='steelblue')
    plt.title('LDA Feature Coefficients — LD1 (Primary Discriminant Axis)')
    plt.xlabel('|Coefficient| — contribution to severity class separation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lda_feature_coefficients.png'))
    plt.close()
    print("Saved lda_feature_coefficients.png")

    # --- Plot: LDA scatter (class separation visualization) ---
    plt.figure(figsize=(10, 7))
    severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    colors = {1: 'red', 2: 'orange', 3: 'steelblue'}
    for sev in sorted(y.unique()):
        mask = y.values == sev
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                    label=severity_labels.get(sev, str(sev)),
                    color=colors.get(sev, 'gray'),
                    alpha=0.3, s=2)
    plt.title('LDA Projection — Class Separation by Accident Severity')
    plt.xlabel(f'LD1 (explains {lda.explained_variance_ratio_[0]*100:.1f}% of between-class variance)')
    plt.ylabel(f'LD2 (explains {lda.explained_variance_ratio_[1]*100:.1f}% of between-class variance)')
    plt.legend(markerscale=5, title='Severity')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lda_scatter.png'))
    plt.close()
    print("Saved lda_scatter.png")

    # Save LDA-transformed data (used optionally in classification phase)
    lda_out = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
    lda_out.to_csv(os.path.join(OUTPUT_DIR, 'X_lda.csv'), index=False)
    print("Saved X_lda.csv")

    # Select top features based on LD1 absolute coefficient magnitude
    # These are the features that most strongly separate severity classes
    top_features = scalings_df_sorted.head(10).index.tolist()
    print(f"\nTop features selected by LDA LD1 coefficients:\n{top_features}")
    X = X[top_features]
    print(f"X reduced to top {len(top_features)} features: {X.shape}")

    # ------------------------------------------------------------------
    # STEP 7 — PCA (for clustering visualization only)
    # ------------------------------------------------------------------
    # PCA is kept here solely to produce a 2D input for clustering (K-Means / DBSCAN).
    # It is NOT used for feature selection — LDA handles that above.
    # PCA is appropriate for clustering because clustering is unsupervised
    # and does not use class labels, which is exactly what PCA is designed for.
    # ------------------------------------------------------------------
    print("\n--- STEP 7: PCA (for clustering visualization) ---")
    scaler_pca = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_pca)
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df.to_csv(os.path.join(OUTPUT_DIR, 'X_pca.csv'), index=False)
    print("Saved X_pca.csv")

    plt.figure(figsize=(10, 7))
    for sev in sorted(y.unique()):
        mask = y.values == sev
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=severity_labels.get(sev, str(sev)),
                    color=colors.get(sev, 'gray'),
                    alpha=0.3, s=2)
    plt.title('PCA 2D Projection — Colored by Accident Severity (used for clustering)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pca_scatter.png'))
    plt.close()
    print("Saved pca_scatter.png")

    # ------------------------------------------------------------------
    # STEP 8 — SAVE FINAL FEATURES
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Save Final Features ---")
    X.to_csv(os.path.join(OUTPUT_DIR, 'X_final.csv'), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, 'y_final.csv'), index=False)
    print(f"Saved X_final.csv — shape: {X.shape}")
    print(f"Saved y_final.csv — shape: {y.shape}")
    print(f"\nFinal selected features: {X.columns.tolist()}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 55)
    print(f"  Method used:          LDA (Linear Discriminant Analysis)")
    print(f"  Reason for LDA:       Maximizes between-class separation")
    print(f"                        for labeled severity classes")
    print(f"  LDA variance (LD1+2): {lda.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  PCA variance (PC1+2): {pca.explained_variance_ratio_.sum()*100:.1f}% (clustering only)")
    print(f"  Features removed:     Month, DayOfWeek, IsWeekend (frequency)")
    print(f"                        Number_of_Casualties (data leakage)")
    print(f"  Final feature count:  {X.shape[1]}")
    print(f"  Final features:       {X.columns.tolist()}")
    print("=" * 55)


if __name__ == "__main__":
    run()