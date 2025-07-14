import joblib
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.losses import MeanSquaredError
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_model():
    """Create a dummy model if real model is missing"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    
    logger.warning("Creating dummy model")
    model = Sequential([
        LSTM(4, input_shape=(24, 4)),
        Dense(4)
    ])
    
    # Create dummy scaler
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0, 0, 0, 0], [100, 50, 100, 10]]))
    
    return model, scaler

def load_model():
    model_path = "models/weather_lstm_model.h5"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler file not found: {scaler_path}")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        logger.warning("Using dummy model - real model or scaler not found")
        return create_dummy_model()
    
    try:
        logger.info("Loading model and scaler")
        model = keras_load_model(
            model_path,
            custom_objects={'mse': MeanSquaredError()}
        )
        scaler = joblib.load(scaler_path)
        logger.info("Successfully loaded trained model")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Falling back to dummy model")
        return create_dummy_model()

def predict_next_24h(last_24h_data, model, scaler):
    logger.info("Generating predictions")
    try:
        # Normalize data
        data_normalized = scaler.transform(last_24h_data)
        seq = data_normalized.copy()
        predictions_normalized = []
        
        # Predict next 24 hours
        for i in range(24):
            input_seq = seq[-24:].reshape(1, 24, 4)
            pred = model.predict(input_seq, verbose=0)[0]
            predictions_normalized.append(pred)
            seq = np.vstack([seq, pred])
            logger.debug(f"Predicted hour {i+1}: {pred}")
        
        # Convert back to original scale
        predictions = scaler.inverse_transform(predictions_normalized)
        logger.info(f"Predictions generated: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return dummy predictions
        return np.zeros((24, 4))
