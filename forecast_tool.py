import requests
from datetime import datetime, timedelta

def get_coordinates(city_name: str):
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(geocode_url, params={"name": city_name})
    response.raise_for_status()
    data = response.json()
    if data.get("results"):
        first = data["results"][0]
        return first["latitude"], first["longitude"]
    else:
        raise ValueError(f"No coordinates found for '{city_name}'")

def get_future_forecast(latitude: float, longitude: float, target_date: str, timezone: str = "auto"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max"],
        "forecast_days": 16,
        "timezone": timezone
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    try:
        date_index = data["daily"]["time"].index(target_date)
        return {
            "temperature": data["daily"]["temperature_2m_mean"][date_index],
            "humidity": data["daily"]["relative_humidity_2m_mean"][date_index],
            "wind_speed": data["daily"]["wind_speed_10m_max"][date_index]
        }
    except:
        return {"error": "Forecast only extends to 16 days"}