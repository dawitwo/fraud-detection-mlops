# monitoring/drift_detector.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH


def simulate_drift_and_monitor():
    reference = pd.read_csv(TRAIN_DATA_PATH)
    current = pd.read_csv(TEST_DATA_PATH)

    # Introduce artificial drift on scaled_amount
    current["scaled_amount"] = (
        current["scaled_amount"] * 3
        + np.random.normal(0, 0.5, len(current))
    )

    report = Report([DataDriftPreset()])
    result = report.run(current, reference)

    result.save_html("monitoring/drift_report.html")

    result_dict = result.dict()
    drift_share = (
        result_dict.get("metrics", [{}])[0]
        .get("result", {})
        .get("share_of_drifted_columns", 0.0)
    )
    print(f"Share of drifted columns: {drift_share:.2%}")

    return drift_share


if __name__ == "__main__":
    share = simulate_drift_and_monitor()
    if share > 0.5:
        print("ALERT: Significant drift detected. Triggering retraining.")