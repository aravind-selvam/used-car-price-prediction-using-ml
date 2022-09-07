import os
import sys

from carprice.exception import CarException
from carprice.util.util import load_object
import pandas as pd


class CarPriceData:

    def __init__(self,
                car_name: str,
                vehicle_age: int,
                km_driven: int,
                seller_type: str,
                fuel_type: str,
                transmission_type: str,
                mileage: float,
                engine: int,
                max_power: float,
                seats: int,
                selling_price: int = None
                 ):
        try:
            self.car_name = car_name
            self.vehicle_age = vehicle_age
            self.km_driven = km_driven
            self.seller_type = seller_type
            self.fuel_type = fuel_type
            self.transmission_type = transmission_type
            self.mileage = mileage
            self.engine = engine
            self.max_power = max_power
            self.seats = seats
            self.selling_price = selling_price
        except Exception as e:
            raise CarException(e, sys) from e

    def get_carprice_input_data_frame(self):

        try:
            carprice_input_dict = self.get_car_data_as_dict()
            return pd.DataFrame(carprice_input_dict)
        except Exception as e:
            raise CarException(e, sys) from e

    def get_car_data_as_dict(self):
        try:
            input_data = {
                "car_name": [self.car_name],
                "vehicle_age": [self.vehicle_age],
                "km_driven": [self.km_driven],
                "seller_type": [self.seller_type],
                "fuel_type": [self.fuel_type],
                "transmission_type": [self.transmission_type],
                "mileage": [self.mileage],
                "engine": [self.engine],
                "max_power": [self.max_power],
                "seats": [self.seats]
                }
            input_data =pd.DataFrame(input_data)
            return input_data
        except Exception as e:
            raise CarException(e, sys)


class CarPricePredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CarException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise CarException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            selling_price_pred = model.predict(X)
            return selling_price_pred
        except Exception as e:
            raise CarException(e, sys) from e