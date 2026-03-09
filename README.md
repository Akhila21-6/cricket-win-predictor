Cricket Match Win Predictor 🏏
An AI-powered win probability predictor that uses Deep Learning (LSTM) to analyze live match situations and provide real-time winning chances for the batting team.

🚀 Project Overview
This project was developed as a final-year B.Tech Computer Science project. It transitions from raw cricket data to a deployed web dashboard, covering the entire Machine Learning lifecycle:

Data Engineering: Processing raw Cricsheet ball-by-ball CSV data into a temporal format.

Deep Learning: Implementing a Sequential Long Short-Term Memory (LSTM) network to capture the "momentum" of a cricket match.

Web Deployment: An interactive dashboard built with Streamlit for real-time predictions.


🛠️ Tech Stack
Language: Python 3.12

Deep Learning: TensorFlow / Keras

Data Analysis: Pandas, NumPy, Scikit-learn

Frontend/UI: Streamlit


📁 Project Structure

cricket-win-predictor/

├── app/

│   └── streamlit_app.py      # Dashboard UI

├── data/

│   ├── raw/                  # Original CSV files

│   └── processed/            # Cleaned data

├── models/

│   └── cricket_win_model.h5  # Trained LSTM model

├── src/

│   ├── data_pipeline.py      # Data cleaning script

│   ├── train.py              # Model training script

│   └── model.py              # LSTM Architecture

└── README.md



⚙️ How to Run
Install Dependencies:

Bash
pip install pandas scikit-learn tensorflow streamlit
Process Data:

Bash
python src/data_pipeline.py
Train the Model:

Bash
python src/train.py
Launch Dashboard:

Bash
streamlit run app/streamlit_app.py



📊 Model Features

The model predicts win probability based on:

Current Score

Wickets Fallen

Balls Bowled (Derived from Overs)

Required Run Rate (Inferred by the LSTM)
