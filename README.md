# Soccer Score Predictor

This project is a simple web application that predicts the outcome of La Liga soccer matches based on historical data. It utilizes machine learning techniques to make predictions and Flask for web development.

## Features

- Predicts the outcome (win, lose, or draw) of La Liga soccer matches.
- Provides explanations for the predictions.
- Web interface for users to input match details and get predictions.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Soccer-Score-Prediction.git
cd Soccer-Score-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```bash
python laliga.py
```

2. Open your web browser and go to http://localhost:5000

3. Input the details of the match (home team, away team, goals scored), and click "Predict" to get the prediction.

OR

1. Click this link

[Soccer-Score-Prediction](https://soccer-score-prediction.onrender.com)

2. Input the details of the match (home team, away team, goals scored), and click "Predict" to get the prediction.

## Dataset

The project uses historical La Liga match data from 1995-2023 stored in the static/la_liga_matches.csv file. This dataset is used to train the machine learning model for making predictions.
