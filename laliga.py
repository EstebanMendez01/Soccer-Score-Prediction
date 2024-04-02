import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the data
data = pd.read_csv('static/la_liga_matches.csv')

# Drop unnecessary columns
data.drop(['Season', 'Date'], axis=1, inplace=True)

# Convert FTR (Full Time Result) to numerical values
data['FTR'] = data['FTR'].map({'H': 1, 'D': 0, 'A': -1})

# Encode categorical variables
label_encoder = LabelEncoder()
data['HomeTeam'] = label_encoder.fit_transform(data['HomeTeam'])
data['AwayTeam'] = label_encoder.transform(data['AwayTeam'])

# Handle missing values for numerical columns
numerical_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG']
imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

# Prepare features and target
X = data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG']]
y = data['FTR']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'soccer_score_prediction_model.pkl')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

def predict_winner(home_team, away_team, fthg, ftag, hthg, htag):
    # Load the trained model
    model = joblib.load('soccer_score_prediction_model.pkl')

    # Load the label encoder
    label_encoder = joblib.load('label_encoder.pkl')

    # Transform input data using the loaded label encoder
    home_team_encoded = label_encoder.transform([home_team])[0]
    away_team_encoded = label_encoder.transform([away_team])[0]

    # Make prediction
    prediction = model.predict([[home_team_encoded, away_team_encoded, fthg, ftag, hthg, htag]])[0]

    # Determine winner
    if prediction > 0:
        winner = home_team
    elif prediction < 0:
        winner = away_team
    else:
        winner = "Draw"

    # Format score
    score = f"{home_team} {fthg}-{ftag} {away_team}"

    # Explanation
    if prediction > 0:
        explanation = f"The model predicts that {home_team} will win."
    elif prediction < 0:
        explanation = f"The model predicts that {away_team} will win."
    else:
        explanation = "The model predicts a draw."

    return prediction, winner, score, explanation

@app.route('/')
def home():
    # Get unique team names for home and away teams
    home_teams = label_encoder.inverse_transform(data['HomeTeam'].unique())
    away_teams = label_encoder.inverse_transform(data['AwayTeam'].unique())
    return render_template('index.html', home_teams=home_teams, away_teams=away_teams)


@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    fthg = int(request.form['fthg'])
    ftag = int(request.form['ftag'])
    hthg = int(request.form['hthg'])
    htag = int(request.form['htag'])

    prediction, winner, score, explanation = predict_winner(home_team, away_team, fthg, ftag, hthg, htag)
    
    # Get unique team names for home and away teams
    home_teams = label_encoder.inverse_transform(data['HomeTeam'].unique())
    away_teams = label_encoder.inverse_transform(data['AwayTeam'].unique())

    return render_template('index.html', prediction=prediction, winner=winner, score=score, explanation=explanation, home_teams=home_teams, away_teams=away_teams)


if __name__ == '__main__':
    app.run(debug=False)
