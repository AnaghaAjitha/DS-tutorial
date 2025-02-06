import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
temperature = np.random.randint(10, 40, 200)  
humidity = np.random.randint(30, 100, 200)    
wind_speed = np.random.randint(0, 30, 200)    

# Define rain conditions: Higher humidity + lower temperature = higher chance of rain
rain = ((humidity - temperature + wind_speed) > 50).astype(int)

df = pd.DataFrame({'temperature': temperature, 'humidity': humidity, 'wind_speed': wind_speed, 'rain': rain})

X = df[['temperature', 'humidity', 'wind_speed']]
y = df['rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict weather conditions (Temp:22°C, Humidity:85%, Wind Speed:10 km/h
new_weather = pd.DataFrame({'temperature': [22], 'humidity': [85], 'wind_speed': [10]})
prediction = model.predict(new_weather)
result = "Rain" if prediction[0] == 1 else "No Rain"

print(f"Prediction for Temp: 22°C, Humidity: 85%, Wind Speed: 10 km/h is that will {result}")
