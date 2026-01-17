import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  r2_score, mean_squared_error, mean_absolute_error

city_attr = pd.read_csv("dataset/city_attributes.csv")
humidity = pd.read_csv("dataset/humidity.csv")
pressure = pd.read_csv("dataset/pressure.csv")
temperature = pd.read_csv("dataset/temperature.csv")
weather_desc = pd.read_csv("dataset/weather_description.csv")
wind_dir = pd.read_csv("dataset/wind_direction.csv")
wind_speed = pd.read_csv("dataset/wind_speed.csv")

