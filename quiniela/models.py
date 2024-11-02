# Standard library imports
import os
import sys
import pickle
import sqlite3

# Third-party imports
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from operator import le
import logging
import time


# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# Project-specific imports
from utils import quiniela_format
from preprocessing import (
    _process_scores,
    _calculate_match_results,
    _normalize_dates,
    _process_seasons,
)
from data_io import load_historical_data, load_matchday
from features import (
    inform_relatives_points,
    inform_win_lost_index,
    last5index,
    last_season_position,
)
from validate import analyze_model_performance
import settings


class QuinielaModel:
    """
    A model for processing and analyzing soccer match data, focused on match results and scoring.

    This class includes methods for preprocessing data, such as score parsing, result classification,
    and date normalization for soccer seasons spanning multiple years.
    """

    FEATURES = [
        "win_punct",
        "lost_punct",
        "points_relative_index",
        "last5_home",
        "last5_away",
        "last_season_points_home",
        "last_season_points_away",
    ]
    TARGET = "result"

    def __init__(self):
        """Initialize the QuinielaModel."""
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input DataFrame by cleaning and transforming data.

        :param df: Input DataFrame containing match data
        :return: Preprocessed DataFrame with additional features
        :raises ValueError: If required columns are missing
        """
        processed_df = df.copy()
        processed_df = _process_scores(processed_df)
        processed_df = _calculate_match_results(processed_df)
        processed_df = _normalize_dates(processed_df)
        processed_df = _process_seasons(processed_df)
        return processed_df

    def calculate_features(
        self, df: pd.DataFrame, start_season, nseasons, index_depth=20, train=True
    ) -> pd.DataFrame:
        """
        Calculate features for the input DataFrame.

        :param df: Input DataFrame containing match data
        :param start_season: Initial season to consider for feature calculation
        :param nseasons: Number of seasons to consider for feature calculation
        :param index_depth: Depth of the index to consider for win and loss calculations
        :return: DataFrame with calculated features
        """
        if index_depth > nseasons:
            index_depth = nseasons

        if train:
            df_train = df.loc[
                (df["season"] > (start_season - nseasons))
                & (df["season"] <= start_season)
            ].copy()
        else:
            df_train = df.loc[
                (df["season"] > (start_season - nseasons))
                & (df["season"] < start_season)
            ].copy()
        logging.info(f"Calculating features for {len(df_train)} matches")
        start_time = time.time()
        logging.info("Calculating relative points")
        df_train = inform_relatives_points(df, df_train)
        logging.info(
            f"Relative points calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating win and loss index")
        df_train = inform_win_lost_index(df, df_train, index_depth)
        logging.info(
            f"Win and loss index calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating last 5 index")
        df_train = last5index(df, df_train)
        logging.info(
            f"Last 5 index calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating last season position")
        df_train = last_season_position(df, df_train)
        logging.info(
            f"Last season position calculated in {time.time() - start_time:.2f} seconds"
        )

        return df_train

    def train(self, df_train: pd.DataFrame, model_name="my_quiniela.model"):
        """Train the model on provided data."""

        x_train = df_train[self.FEATURES]
        y_train = df_train[self.TARGET]

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=40
        )

        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)

        self.save(model_name)
        logging.info(f"Model saved as {model_name}")

        return clf, x_val, y_val

    def validate(self, clf, x_val: pd.DataFrame, y_val: pd.DataFrame):
        """Validate the model on provided data."""

        feature_importance = pd.DataFrame(
            {
                "feature": self.FEATURES,
                "importance": clf.feature_importances_,
            }
        )
        y_val_pred = clf.predict(x_val)
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )
        analyze_model_performance(feature_importance, y_val, y_val_pred, clf)

    def predict_result(
        self, season: str, matchday: int, depth: int, train=True
    ) -> pd.DataFrame:
        """Generate predictions on the provided data."""

        if train:
            df = load_historical_data(season, depth)
        else:
            df = load_matchday(season, 1, matchday)
        df = self.preprocess(df)
        season = int(season.split("/")[0])
        df_matchday = df.loc[
            (df["season"] == season)
            & (df["matchday"] == matchday)
            & (df["division"] == 1)
        ].copy()
        df_matchday = self.calculate_features(df, df_matchday, season, depth, False)
        x_predict = df_matchday[self.FEATURES]

        clf = self.load("my_quiniela.model")
        y_predict = clf.predict(x_predict)
        y_predict = le.inverse_transform(y_predict)
        df_matchday["prediction"] = y_predict
        df_matchday["correct"] = df_matchday["result"] == df_matchday["prediction"]
        df_matchday = quiniela_format(df_matchday)

        return df_matchday

    @classmethod
    def load(cls, filename):
        """Load model from file."""
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert isinstance(model, cls)
        return model

    def save(self, filename):
        """Save model to a file."""
        with open(os.path.join("..", "models", filename), "wb") as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = QuinielaModel()
    # Un argument ha de ser nseasons
    nseasons = 2
    df = load_historical_data("2004/2005", nseasons)
    logging.info("Data loaded")
    processed_df = model.preprocess(df)
    logging.info("Processed data")
    logging.info("Starting to calculate features")
    df_train = model.calculate_features(
        processed_df, 2004, nseasons
    )  # Revisar arguments q paso
    logging.info("Features calculated")
    logging.info("Starting to train model")
    clf, x_val, y_val = model.train(df_train, "Test.model")
    model.validate(clf, x_val, y_val)

    logging.info("Example prediction")
    df_matchday = model.predict_result("2005/2006", 1, nseasons)
    logging.info(df_matchday)
