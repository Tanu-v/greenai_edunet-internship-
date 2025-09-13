# Configuration file for AQI Monitoring

# Paths
DATA_PATH = "data/processed_aqi.csv"
MODEL_PATH = "models/aqi_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoders.pkl"   # optional if saving encoders too

# Model training parameters
RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2
