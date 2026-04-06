# main.py
# Run this to execute the entire pipeline end to end.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import preprocessing
import feature_selection
import classification
import clustering
import regression
import visualizations

if __name__ == "__main__":
    print("=" * 60)
    print("UK TRAFFIC ACCIDENT ANALYSIS — BIG DATA PROJECT")
    print("=" * 60)

    print("\n[1/6] Running Preprocessing...")
    preprocessing.run()

    print("\n[2/6] Running Feature Selection...")
    feature_selection.run()

    print("\n[3/6] Running Classification (Expert 1)...")
    classification.run()

    print("\n[4/6] Running Clustering (Expert 2)...")
    clustering.run()

    print("\n[5/6] Running Regression (Expert 3)...")
    regression.run()

    print("\n[6/6] Generating Final Visualizations...")
    visualizations.run()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE. Check the outputs/ folder.")
    print("=" * 60)
