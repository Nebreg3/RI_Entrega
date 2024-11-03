"""
This module provides functions for loading and saving data.
"""
import os
import sys
import sqlite3

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import settings

def load_data():
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql("SELECT * FROM Matches", conn)
    return data

def save_predictions(predictions):
    predictions_dir = os.path.join(os.path.dirname(__file__), "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    predictions_path = os.path.join(predictions_dir, "predictions.csv")
    predictions.to_csv(predictions_path, index=False)
