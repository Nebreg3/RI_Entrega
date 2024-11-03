## LaQuiniela of LaLiga

### Team Members

- **Néstor Bravo Egea** (NIU: 1563318)
- **Juan Manuel Sánchez Melián** (NIU: 1598286)
- **Jose Carlos Sanz Tirado** (NIU: 1742458)
- **Biel Majó Cornet** (NIU: 1568210)

### La Quiniela Model

This repository contains the code and resources for the La Quiniela model, which is used to train, validate, and predict soccer match results in 'La Quiniela' style. The model is based on a GradientBoostingClassifier from scikit-learn.

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

### Installation

This code has been developed using Python 3.7.9.

1. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training the Model

To train the model with historical data, use the `train` subcommand:
```bash
python cli.py train <train_season> <train_nseasons> --model_name <model_name> [--validate] [--depth <depth>]
```
- `train_season`: The starting season for training. Format: YYYY/YYYY.
- `train_nseasons`: The number of previous seasons to use for training.
- `--model_name`: The name to save the trained model with.
- `--validate`: Flag to indicate whether to validate the model after training.
- `--depth`: Depth of the index win_lose calculation.

#### Predicting Match Results

To make predictions for a specific matchday in a season, use the `predict` subcommand:
```bash
python cli.py predict <predict_season> <predict_division> <predict_matchday> --model_name <model_name>
```
- `predict_season`: The season to predict. Format: YYYY/YYYY.
- `predict_division`: The division to predict (1 or 2).
- `predict_matchday`: The matchday to predict.
- `--model_name`: The name of the model to use for prediction.

### Features

- **Preprocessing**: Clean and transform input data.
- **Feature Calculation**: Calculate various features for the model.
- **Model Training**: Train the model using historical data.
- **Model Validation**: Validate the model's performance.
- **Prediction**: Generate predictions for upcoming matches.
