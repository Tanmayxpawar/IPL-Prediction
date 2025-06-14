import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# --- Feature Engineering Functions ---
def create_features(data):
    """
    Applies feature engineering identical to training:
    - is_high_target: 1 if target_runs >= 190 else 0
    - run_wickets_pressure_inn1: target_runs / (wickets_lost_inn1 + 1)
    """
    high_target_threshold = 190
    data['is_high_target'] = (data['target_runs'] >= high_target_threshold).astype(int)
    data['run_wickets_pressure_inn1'] = data['target_runs'] / (data['wickets_lost_inn1'] + 1)
    return data

# --- Input Transformation ---
def transform_input(user_input, label_encoders):
    df = pd.DataFrame([user_input])
    # Create features
    df = create_features(df)
    # Encode categoricals
    mappings = {
        'city': 'cities',
        'venue': 'venues',
        'team1_bat_first': 'teams',
        'team2_chase': 'teams',
        'toss_winner': 'teams'
    }
    for col, key in mappings.items():
        try:
            df[col + '_encoded'] = label_encoders[key].transform(df[col])
        except ValueError:
            df[col + '_encoded'] = 0
        df.drop(col, axis=1, inplace=True)
    # Encode toss decision
    df['toss_decision_encoded'] = (df['toss_decision'] == 'bat').astype(int)
    df.drop('toss_decision', axis=1, inplace=True)
    # Ensure season exists
    if 'season' not in df:
        df['season'] = 2024
    # Select and order features
    features = [
        'target_runs', 'wickets_lost_inn1', 'is_high_target', 'run_wickets_pressure_inn1',
        'city_encoded', 'venue_encoded', 'team1_bat_first_encoded', 'team2_chase_encoded',
        'toss_winner_encoded', 'toss_decision_encoded', 'season'
    ]
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    return df[features]

# --- Streamlit App ---
def main():
    # Page config
    st.set_page_config(page_title="IPL Match Win Predictor", layout="wide")
    st.title("🏏 IPL Match Win Predictor")

    @st.cache_resource
    def load_models_encoders():
        # Load models
        with open('Models/rf_model.pkl', 'rb') as f: rf = pickle.load(f)
        with open('Models/xgb_model.pkl', 'rb') as f: xgb = pickle.load(f)
        with open('Models/lr_pipeline.pkl', 'rb') as f: lr = pickle.load(f)
        # Load encoders
        with open('Models/fitted_label_encoders.pkl', 'rb') as f: enc = pickle.load(f)
        return rf, xgb, lr, enc

    rf_model, xgb_model, lr_pipeline, label_encoders = load_models_encoders()

    # Get categories from encoders
    teams  = list(label_encoders['teams'].classes_)
    cities = list(label_encoders['cities'].classes_)
    venues = list(label_encoders['venues'].classes_)

    # Sidebar inputs
    st.sidebar.header("Enter Match Details")
    team1 = st.sidebar.selectbox("Team Batting First", teams)
    team2 = st.sidebar.selectbox("Team Chasing", teams)
    if team1 == team2:
        st.sidebar.error("Teams must differ.")
        return
    city = st.sidebar.selectbox("City", cities)
    venue = st.sidebar.selectbox("Venue", venues)
    toss_winner   = st.sidebar.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.sidebar.radio("Toss Decision", ['bat', 'field'])
    season = st.sidebar.number_input("Season/Year", 2008, 2030, 2024)
    target_runs = st.sidebar.number_input("Target Runs", 0, 400, 180)
    wickets_lost = st.sidebar.number_input("Wickets Lost (Innings 1)", 0, 10, 6)

    # Prepare input dict
    user_input = {
        'target_runs': target_runs,
        'wickets_lost_inn1': wickets_lost,
        'city': city,
        'venue': venue,
        'team1_bat_first': team1,
        'team2_chase': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'season': season
    }

    if st.sidebar.button("Predict Winner"):
        # Transform features
        df_in = transform_input(user_input, label_encoders)

        # List of features used
        feature_list = [
            'target_runs', 'wickets_lost_inn1', 'is_high_target', 'run_wickets_pressure_inn1',
            'city_encoded', 'venue_encoded', 'team1_bat_first_encoded', 'team2_chase_encoded',
            'toss_winner_encoded', 'toss_decision_encoded', 'season'
        ]

        # Display engineered feature values
        st.subheader("🚀 Engineered Features & Encodings")
        ef_values = {feat: df_in.iloc[0][feat] for feat in feature_list}
        st.json(ef_values)

        # Predict with each model
        cols = st.columns(3)
        preds, probs, names = [], [], ["Random Forest", "XGBoost", "Logistic Regression"]
        models = [rf_model, xgb_model, lr_pipeline]
        for mdl, nm, col in zip(models, names, cols):
            pred = mdl.predict(df_in)[0]
            proba = mdl.predict_proba(df_in)[0]
            winner = team2 if pred == 1 else team1
            conf = max(proba) * 100
            preds.append(pred)
            probs.append(proba)
            with col:
                st.subheader(nm)
                st.write(f"Winner: {winner}")
                st.write(f"Confidence: {conf:.1f}%")
                fig = go.Figure([go.Bar(x=[team1, team2], y=[proba[0]*100, proba[1]*100])])
                fig.update_layout(yaxis_title="Probability (%)")
                st.plotly_chart(fig, use_container_width=True)

        # Ensemble
        avg_proba = np.mean(probs, axis=0)
        final = team2 if avg_proba[1] > avg_proba[0] else team1
        final_conf = max(avg_proba) * 100
        st.markdown(f"## 🎯 Ensemble Prediction: {final} ({final_conf:.1f}% confidence)")

        # Consensus
        winner_names = [team2 if p==1 else team1 for p in preds]
        msg = "All models agree!" if len(set(winner_names))==1 else "Models disagree"
        st.info(msg)

    # Footer explanation
    st.markdown("---")
    st.markdown("This app uses exactly the 11 features from training, with identical encoding and feature engineering.")

if __name__ == '__main__':
    main()
