import streamlit as st
import tensorflow as tf
import numpy as np
import os

st.set_page_config(page_title="Cricket Win Predictor", page_icon="🏏")

st.title("Cricket Match Win Predictor 🏏")
st.write("Enter the current match situation to predict the win probability.")

# Function to load model with caching to prevent blank screens
@st.cache_resource
def load_my_model():
    model_path = "models/cricket_win_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_my_model()

if model is None:
    st.error("Model file not found! Please run 'python src/train.py' first.")
else:
    # Sidebar for inputs to keep the main screen clean
    st.sidebar.header("Match Statistics")
    score = st.sidebar.number_input("Current Score", value=100, min_value=0)
    wickets = st.sidebar.slider("Wickets Fallen", 0, 10, 2)
    balls = st.sidebar.number_input("Balls Bowled (Out of 120)", value=60, min_value=0, max_value=120)

    if st.button("Predict Win Probability"):
        with st.spinner('Calculating...'):
            # Features: current_score, wickets_remaining, balls_remaining, over, ball
            wickets_rem = 10 - wickets
            balls_rem = 120 - balls
            over_num = balls // 6
            ball_num = balls % 6
            
            input_data = np.array([[score, wickets_rem, balls_rem, over_num, ball_num]])
            input_data = input_data.reshape((1, 1, 5))
            
            prediction = model.predict(input_data)[0][0]
            
            st.subheader(f"Probability of Batting Team Winning: {prediction*100:.2f}%")
            st.progress(float(prediction))