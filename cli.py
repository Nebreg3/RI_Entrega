#!/usr/bin/env python
import logging
import argparse
from datetime import datetime

import settings
from quiniela import models, data_io

parser = argparse.ArgumentParser()
task_subparser = parser.add_subparsers(help='Task to perform', dest='task')
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
    
    # Ensure args.task is not None
    if args.task is None:
        parser.print_help()
        exit(1)
    
    log_filename = settings.LOGS_PATH / f"{args.task}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
    )
    
    if args.task == "train":
        logging.info(f"Training LaQuiniela model starting in seasons {args.train_season} with {args.train_nseasons} previous seasons")
        model = models.QuinielaModel()
        training_data = data_io.load_historical_data(args.train_season, args.train_nseasons)
        logging.info("Data loaded")
        processed_df = model.preprocess(training_data)
        logging.info("Processed data")
        logging.info("Starting to calculate features")
        df_features = model.calculate_features(
            processed_df, args.train_season, args.train_nseasons
        ) 
        logging.info("Features calculated")
        logging.info("Starting to train model")
        clf, x_val, y_val = model.train(df_features)
        model.save(settings.MODELS_PATH / args.model_name)
        logging.info(f"Model successfully trained and saved in {settings.MODELS_PATH / args.model_name}")
        if args.validate:
            model.validate(clf, x_val, y_val)

    elif args.task == "predict":
        logging.info(f"Predicting matchday {args.predict_matchday} in season {args.predict_season}, division {args.predict_division}")
        model = models.QuinielaModel.load(settings.MODELS_PATH / args.model_name)
        predict_data = data_io.load_matchday(args.predict_season, args.predict_division, args.predict_matchday)
        predict_data["pred"] = model.predict(predict_data)
        print(f"Matchday {args.predict_matchday} - LaLiga - Division {args.predict_division} - Season {args.predict_season}")
        print("=" * 70)
        for _, row in predict_data.iterrows():
            print(f"{row['home_team']:^30s} vs {row['away_team']:^30s} --> {row['pred']}")
        data_io.save_predictions(predict_data)