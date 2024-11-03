"""
PYTHON VERSION USED: 3.7.9
This script is the entry point for the command-line interface (CLI) of the LaQuiniela model. It uses the argparse module to define the command-line arguments and subcommands for training and predicting with the model.

The script defines two subcommands: train and predict. The train subcommand is used to train the model with historical data, while the predict subcommand is used to make predictions for a specific matchday in a season.

The train subcommand takes the following arguments:
- train_season: The starting season for training.
- train_nseasons: The number of previous seasons to use for training.
- model_name: The name to save the trained model with.
- validate: Flag to indicate whether to validate the model after training.
- depth: Depth of the index win_lose calculation.

The predict subcommand takes the following arguments:
- predict_season: The season to predict.
- predict_division: The division to predict (1 or 2).
- predict_matchday: The matchday to predict.
- model_name: The name of the model to use for prediction.

An example usage of the CLI would be:
```
python cli.py train 2010/2011 10 --model_name my_quiniela.model --validate --depth 20
```
or
```
python cli.py predict 2019/2020 1 10 --model_name my_quiniela.model
```
"""

#!/usr/bin/env python
import logging
import argparse
from datetime import datetime

import settings
from quiniela import models, data_io

parser = argparse.ArgumentParser()
task_subparser = parser.add_subparsers(help="Task to perform", dest="task")
train_parser = task_subparser.add_parser("train")
train_parser.add_argument(
    "train_season",
    help="Season to start training from. Format: YYYY/YYYY",
)
train_parser.add_argument(
    "train_nseasons",
    type=int,
    default=20,
    help="Number of seasons to use for training. Starting from season provided.",
)
train_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name to save the model with.",
)
train_parser.add_argument(
    "--validate",
    action="store_true",
    help="Validate the model after training.",
)
train_parser.add_argument(
    "--depth",
    type=int,
    default=20,
    help="Depth of index win_lose calculation.",
)

predict_parser = task_subparser.add_parser("predict")
predict_parser.add_argument(
    "predict_season",
    help="Season to predict",
)
predict_parser.add_argument(
    "predict_division",
    type=int,
    choices=[1, 2],
    help="Division to predict (either 1 or 2)",
)
predict_parser.add_argument(
    "predict_matchday",
    type=int,
    help="Matchday to predict",
)
predict_parser.add_argument(
    "--model_name",
    default="my_quiniela.model",
    help="The name of the model you want to use.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.task is None:
        parser.print_help()
        exit(1)

    log_filename = (
        settings.LOGS_PATH
        / f"{args.task}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
    )

    if args.task == "train":
        logging.info(
            f"Training LaQuiniela model starting in seasons {args.train_season} with {args.train_nseasons} previous seasons"
        )
        model = models.QuinielaModel()

        all_data = data_io.load_data()
        logging.info("Data loaded")

        processed_df = model.preprocess(all_data)
        args.train_season = int(args.train_season.split("/")[0])
        training_data = processed_df.loc[
            (processed_df["season"] > (args.train_season - args.train_nseasons))
            & (processed_df["season"] <= args.train_season)
        ].copy()
        logging.info("Data processed")

        logging.info("Starting to calculate features")
        df_features = model.calculate_features(processed_df, training_data, args.depth)
        logging.info("Features calculated")

        logging.info("Starting to train model")
        clf, x_val, y_val = model.train(df_features)

        model.save(settings.MODELS_PATH / args.model_name)
        logging.info(
            f"Model successfully trained and saved in {settings.MODELS_PATH / args.model_name}.pkl"
        )
        if args.validate:
            model.validate(clf, x_val, y_val)

    elif args.task == "predict":
        logging.info(
            f"Predicting matchday {args.predict_matchday} in season {args.predict_season}, division {args.predict_division}"
        )
        model = models.QuinielaModel.load(settings.MODELS_PATH / args.model_name)

        all_data = data_io.load_data()
        logging.info("Data loaded")

        processed_df = model.preprocess(all_data)
        args.predict_season = int(args.predict_season.split("/")[0])
        predict_data = processed_df.loc[
            (processed_df["season"] == args.predict_season)
            & (processed_df["matchday"] == args.predict_matchday)
            & (processed_df["division"] == args.predict_division)
        ].copy()
        logging.info("Data processed")

        logging.info("Starting to calculate features")
        df_features = model.calculate_features(processed_df, predict_data)
        logging.info("Features calculated")

        logging.info("Making predictions")
        predict_data = model.predict(df_features)
        logging.info("Predictions made")
        data_io.save_predictions(predict_data)

        logging.info(
            f"Matchday {args.predict_matchday} - LaLiga - Division {args.predict_division} - Season {args.predict_season}"
        )
        logging.info("=" * 70)
        for _, row in predict_data.iterrows():
            logging.info(
                f"{row['home_team']:^30s} vs {row['away_team']:^30s} --> {row['prediction']}"
            )
