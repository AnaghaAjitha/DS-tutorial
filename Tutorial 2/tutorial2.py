import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42) #using seed ensures the same random numbers are generated every time the code runs for fiar model evaluation
temperature = np.random.randint(10, 40, 200)#generate random temperatures values between 10 and 40 celsius 
humidity = np.random.randint(30, 100, 200)#generate random humidity levels between 30% and 100%  
wind_speed = np.random.randint(0, 30, 200)#generate random wind speeds between 0 and 30 km/h   

#condition for rain: Higher humidity + lower temperature = higher chance of rain
rain = ((humidity - temperature + wind_speed) > 50).astype(int)

#creating a pandas DataFrame to store generated data 
df = pd.DataFrame({'temperature': temperature, 'humidity': humidity, 'wind_speed': wind_speed, 'rain': rain})

#X-input; y-target variable 
X = df[['temperature', 'humidity', 'wind_speed']]
y = df['rain']

#divide dataset into training-80% and test-20% data so as to evaluate performance of model; using sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression() #initializing a logistic regression model from sklearn  
model.fit(X_train, y_train)#training the model using the training data  

y_pred = model.predict(X_test)#predict rain chances on the test data  
accuracy = accuracy_score(y_test, y_pred)#evaluate model performance; using sklearn  

#print result
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#predict weather conditions (temperature-22C, humidity-85%, wind speed:10 km/h)
new_weather = pd.DataFrame({'temperature': [22], 'humidity': [85], 'wind_speed': [10]})

#predict whether it will rain or not 
prediction = model.predict(new_weather)
result = "Rain" if prediction[0] == 1 else "No Rain"

#display the prediction result
print(f"Prediction for Temp: 22°C, Humidity: 85%, Wind Speed: 10 km/h is that will {result}")
