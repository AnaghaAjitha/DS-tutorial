## DS Tutorial 2
Implemented a Logistic Regression model to predict whether it will rain or not based on weather conditions such as temperature, humidity, and wind speed. The dataset is generated synthetically using NumPy.

# Dataset

The dataset consists of 200 randomly generated records with the following features:
Temperature (°C): Ranges from 10°C to 40°C.
Humidity (%): Ranges from 30% to 100%.
Wind Speed (km/h): Ranges from 0 to 30 km/h.
Rain (Target Variable): 1 if it will rain, 0 if it will not.
The condition for rain is defined as: (humidity-temperature+wind_speed)>50

# Output

Model Accuracy: The performance of the logistic regression model.
Confusion Matrix: A matrix showing correct and incorrect predictions.
Rain Prediction: The model predicts whether it will rain based on given input values.

Output for the code
Model Accuracy: 1.00
Confusion Matrix:
 [[14  0]
 [ 0 26]]
Prediction for Temp: 22°C, Humidity: 85%, Wind Speed: 10 km/h is that will Rain
