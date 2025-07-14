from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class WeatherData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    datetime = db.Column(db.DateTime, nullable=False, unique=True)
    humidity = db.Column(db.Float, nullable=False)
    temp = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    wind_dir = db.Column(db.String(10), nullable=True)
    rain = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<WeatherData {self.id}: {self.datetime} - Temp: {self.temp}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    forecast_datetime = db.Column(db.DateTime, nullable=False, unique=True)
    humidity = db.Column(db.Float, nullable=False)
    temp = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    rain = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.forecast_datetime} - Temp: {self.temp}>'
