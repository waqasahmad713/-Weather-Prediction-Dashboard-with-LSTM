from flask import Flask, request, jsonify, render_template
from database import db, WeatherData, Prediction
from weather_model import load_model, predict_next_24h
from datetime import datetime, timedelta
import numpy as np
import json
import os
import csv
from io import TextIOWrapper
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///weather.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
try:
    model, scaler = load_model()
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model, scaler = None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    try:
        dt = datetime.fromisoformat(data['datetime'])
    except:
        return jsonify({"error": "Invalid datetime format"}), 400
    
    # Save to database
    new_data = WeatherData(
        datetime=dt,
        humidity=data['humidity'],
        temp=data['temp'],
        wind_speed=data['wind_speed'],
        wind_dir=data.get('wind_dir', ''),
        rain=data['rain']
    )
    db.session.add(new_data)
    db.session.commit()
    
    # Check if we have enough data for prediction
    if WeatherData.query.count() >= 24:
        # Get last 24 records
        last_24 = WeatherData.query.order_by(WeatherData.datetime.desc()).limit(24).all()
        last_24_sorted = sorted(last_24, key=lambda x: x.datetime)
        
        # Prepare data for prediction
        data_array = np.array([[
            d.humidity, 
            d.temp, 
            d.wind_speed, 
            d.rain
        ] for d in last_24_sorted])
        
        # Make predictions
        predictions = predict_next_24h(data_array, model, scaler)
        
        # Save predictions to database
        Prediction.query.delete()  # Clear old predictions
        for i, pred in enumerate(predictions):
            forecast_time = last_24_sorted[-1].datetime + timedelta(hours=i+1)
            new_pred = Prediction(
                forecast_datetime=forecast_time,
                humidity=pred[0],
                temp=pred[1],
                wind_speed=pred[2],
                rain=pred[3]
            )
            db.session.add(new_pred)
        db.session.commit()
        
        return jsonify({"message": "Data received and predictions updated"}), 201
    
    return jsonify({"message": "Data received"}), 201

@app.route('/api/weather', methods=['GET'])
def get_weather():
    try:
        current = WeatherData.query.order_by(WeatherData.datetime.desc()).first()
        predictions = Prediction.query.order_by(Prediction.forecast_datetime.asc()).all()
        print("currentcurrent", current)
        print("predictionspredictions", predictions)
        if not current:
            logger.warning("No current weather data available")
            return jsonify({"error": "No weather data available"}), 404
        
        logger.debug(f"Current data: {current}")
        logger.debug(f"Found {len(predictions)} predictions")
        
        return jsonify({
            "current": {
                "datetime": current.datetime.isoformat(),
                "temp": current.temp,
                "humidity": current.humidity,
                "wind_speed": current.wind_speed,
                "wind_dir": current.wind_dir,
                "rain": current.rain
            },
            "predictions": [{
                "datetime": p.forecast_datetime.isoformat(),
                "temp": p.temp,
                "humidity": p.humidity,
                "wind_speed": p.wind_speed,
                "rain": p.rain
            } for p in predictions]
        })
    except Exception as e:
        logger.error(f"Error in /api/weather: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data-logs', methods=['GET'])
def get_data_logs():
    logs = WeatherData.query.order_by(WeatherData.datetime.desc()).limit(50).all()
    return jsonify([{
        "datetime": log.datetime.isoformat(),
        "temp": log.temp,
        "humidity": log.humidity,
        "wind_speed": log.wind_speed,
        "rain": log.rain
    } for log in logs])

@app.route('/api/predict', methods=['POST'])
def manual_predict():
    # Get last 24 records
    last_24 = WeatherData.query.order_by(WeatherData.datetime.desc()).limit(24).all()
    print("last_24last_24last_24last_24", last_24)
    if len(last_24) < 24:
        return jsonify({"message": "Not enough data (need 24 records)"}), 400
    
    last_24_sorted = sorted(last_24, key=lambda x: x.datetime)
    data_array = np.array([[d.humidity, d.temp, d.wind_speed, d.rain] for d in last_24_sorted])
    
    predictions = predict_next_24h(data_array, model, scaler)
    
    # Save predictions to database
    Prediction.query.delete()
    for i, pred in enumerate(predictions):
        forecast_time = last_24_sorted[-1].datetime + timedelta(hours=i+1)
        new_pred = Prediction(
            forecast_datetime=forecast_time,
            humidity=pred[0],
            temp=pred[1],
            wind_speed=pred[2],
            rain=pred[3]
        )
        db.session.add(new_pred)
    db.session.commit()
    
    return jsonify({"message": "Manual prediction completed successfully"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_json_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.json'):
        try:
            data = json.load(file)
            # Process each entry in the JSON file
            for entry in data:
                dt = datetime.fromisoformat(entry['datetime'])
                print("entry", entry)  # Print the entry to view its contents
                new_data = WeatherData(
                    datetime=dt,
                    humidity=entry['humidity'],
                    temp=entry['temp'],
                    wind_speed=entry['wind_speed'],
                    wind_dir=entry.get('wind_dir', ''),
                    rain=entry['rain']
                )
                print("new data", new_data)  # Print the entry to view its contents

                db.session.add(new_data)
            db.session.commit()
            return jsonify({"message": f"{len(data)} records uploaded successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Only JSON files are accepted"}), 400

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # Process CSV file
            csv_file = TextIOWrapper(file, encoding='utf-8')
            reader = csv.DictReader(csv_file)
            
            records = []
            for row in reader:
                try:
                    # Handle different datetime formats
                    dt_formats = [
                        "%Y-%m-%dT%H:%M:%S",   # ISO format
                        "%Y-%m-%d %H:%M:%S",    # SQL format
                        "%m/%d/%Y %H:%M",       # US format
                        "%d/%m/%Y %H:%M"        # EU format
                    ]
                    
                    dt = None
                    for fmt in dt_formats:
                        try:
                            dt = datetime.strptime(row['datetime'], fmt)
                            break
                        except ValueError:
                            continue
                    
                    if not dt:
                        raise ValueError(f"Unrecognized datetime format: {row['datetime']}")
                    
                    records.append(WeatherData(
                        datetime=dt,
                        humidity=float(row['humidity']),
                        temp=float(row['temp']),
                        wind_speed=float(row['wind_speed']),
                        wind_dir=row.get('wind_dir', ''),
                        rain=float(row.get('rain', 0))
                    ))
                except Exception as e:
                    logger.error(f"Error processing row: {row} - {e}")
            
            # Batch insert records
            db.session.bulk_save_objects(records)
            db.session.commit()
            
            return jsonify({
                "message": f"Successfully uploaded {len(records)} records from CSV",
                "processed": len(records)
            }), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Only CSV files are accepted"}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
