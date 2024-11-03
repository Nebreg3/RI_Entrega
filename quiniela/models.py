import os
import sys
import pickle
import logging
import time
from operator import le

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .utils import quiniela_format
from .preprocessing import (
    _process_scores,
    _calculate_match_results,
    _normalize_dates,
    _process_seasons,
)
from .features import (
    inform_relatives_points,
    inform_win_lost_index,
    last5index,
    last_season_position,
)
from .validate import analyze_model_performance


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
        self, df: pd.DataFrame, df_features, depth: int = float('inf')
    ) -> pd.DataFrame:
        """
        Calculate features for the input DataFrame.

        :param df: Input DataFrame containing match data
        :param df_features: DataFrame containing data to calculate features for
        :param depth: Depth for calculating features, default is infinity
        :return: DataFrame with calculated features
        """

        logging.info(f"Calculating features for {len(df)} matches")
        start_time = time.time()
        logging.info("Calculating relative points")
        df_features = inform_relatives_points(df, df_features)
        logging.info(
            f"Relative points calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating win and loss index")
        df_features = inform_win_lost_index(df, df_features, depth)
        logging.info(
            f"Win and loss index calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating last 5 index")
        df_features = last5index(df, df_features)
        logging.info(
            f"Last 5 index calculated in {time.time() - start_time:.2f} seconds"
        )

        start_time = time.time()
        logging.info("Calculating last season position")
        df_features = last_season_position(df, df_features)
        logging.info(
            f"Last season position calculated in {time.time() - start_time:.2f} seconds"
        )

        return df_features

    def train(self, df_train: pd.DataFrame):
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

    def predict(self, df_matchday: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions on the provided data.
        
        :param df_matchday: DataFrame with matchday data
        
        :return: DataFrame with predictions
        """

        x_predict = df_matchday[self.FEATURES]

        y_predict = self.model.predict(x_predict)
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


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     model = QuinielaModel()
#     # Un argument ha de ser nseasons
#     nseasons = 2
#     df = load_historical_data("2004/2005", nseasons)
#     logging.info("Data loaded")
#     processed_df = model.preprocess(df)
#     logging.info("Processed data")
#     logging.info("Starting to calculate features")
#     df_train = model.calculate_features(
#         processed_df, 2004, nseasons
#     )  # Revisar arguments q paso
#     logging.info("Features calculated")
#     logging.info("Starting to train model")
#     clf, x_val, y_val = model.train(df_train)
#     model.save(model_name)
#     logging.info(f"Model saved as {model_name}")
#     model.validate(clf, x_val, y_val)

#     logging.info("Example prediction")
#     df_matchday = model.predict_result("2005/2006", 1, nseasons)
#     logging.info(df_matchday)
