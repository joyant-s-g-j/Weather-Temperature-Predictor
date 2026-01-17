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

# Load datasets
city_attr = pd.read_csv("dataset/city_attributes.csv")
humidity = pd.read_csv("dataset/humidity.csv")
pressure = pd.read_csv("dataset/pressure.csv")
temperature = pd.read_csv("dataset/temperature.csv")
weather_desc = pd.read_csv("dataset/weather_description.csv")
wind_dir = pd.read_csv("dataset/wind_direction.csv")
wind_speed = pd.read_csv("dataset/wind_speed.csv")

# Convert wide to long format
def melt_df(df, val_name):
  return df.melt(id_vars=['datetime'], var_name='city', value_name=val_name)

temp = melt_df(temperature, "temperature")
humi = melt_df(humidity, "humidity")
pres = melt_df(pressure, "pressure")
wthr_desc = melt_df(weather_desc, "weather_description")
wind_d = melt_df(wind_dir, "wind_direction")
wind_s = melt_df(wind_speed, "wind_speed")

# Merge datasets
df = temp.merge(humi, on=['datetime', 'city']) \
         .merge(pres, on=['datetime', 'city']) \
         .merge(wthr_desc, on=['datetime', 'city']) \
         .merge(wind_d, on=['datetime', 'city']) \
         .merge(wind_s, on=['datetime', 'city'])

# Convert temperature from Kelvin to Celsius and remove outliers
df['temperature'] = df['temperature'] - 273.15
df = df[df['temperature'].between(-50, 60)]

# Split features and target
X = df.drop(columns=['temperature','datetime','city'], axis=1)
y = df['temperature']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers= [
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# Model pipeline
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Evaluation metrics
y_pred = pipeline.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Save the trained model
with open("weather_temp_model.pkl", "wb") as file:
  pickle.dump(pipeline, file)

print("Model saved as weather_temp_model.pkl")