# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Reshape
from keras.losses import Huber
# Data Preprocessing
dataset_train = pd.read_csv("D:\\project final year\\train_data.csv")

# Filling missing values with mean using imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset_train[['Weight', 'Calories intake']] = imputer.fit_transform(
    dataset_train[['Weight', 'Calories intake']]
)

# Scaling the dataset
sc = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = sc.fit_transform(dataset_train[['Weight', 'Calories intake']])

# Creating input and output for training
x_train = []
y_train = []
time_steps = 7

for i in range(time_steps, len(scaled_training_set) - time_steps):
    x_train.append(scaled_training_set[i-time_steps:i, :])  # Last 7 days
    y_train.append(scaled_training_set[i:i+time_steps, :])  # Next 7 days

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshaping the data to match LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
    
    # Building the Recurrent Neural Network (RNN)
    regressor = Sequential()
    
    # First LSTM layer
    regressor.add(LSTM(units=600, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    regressor.add(Dropout(0.1))
    
    # Second LSTM layer
    regressor.add(LSTM(units=600, return_sequences=True))
    regressor.add(Dropout(0.1))
    
    # Third LSTM layer
    regressor.add(LSTM(units=600, return_sequences=True))
    regressor.add(Dropout(0.1))
    
    # Fourth LSTM layer
    regressor.add(LSTM(units=600, return_sequences=False))
    regressor.add(Dropout(0.1))
    
    
    
    # Output layer (predicting 7 days, 2 features)
    regressor.add(Dense(units=7 * 2))  # 7 time steps, 2 features (Weight and Calories)
    regressor.add(Reshape((7, 2)))  # Reshape output to match the expected target shape
    
    # Compile the model
    regressor.compile(optimizer='rmsprop', loss=Huber())
    
    # Fit the model
    regressor.fit(x_train, y_train, epochs=30, batch_size=15)

# Making the prediction and visualizing the result

# Load the test dataset
dataset_test = pd.read_csv("D:\\project final year\\test_data.csv")

# Combine the train and test datasets
dataset_total = pd.concat((dataset_train[['Weight', 'Calories intake']], dataset_test[['Weight', 'Calories intake']]), axis=0)

# Extract the last 7 days of data for prediction
inputs = dataset_total[len(dataset_total) - 7:].values  # Last 7 days of data

# Reshape the inputs to match the model's expected shape (1 sample, 7 time steps, 2 features)
inputs_scaled = sc.transform(inputs.reshape(-1, 2))  # Flatten the input to apply scaler
inputs_scaled = inputs_scaled.reshape(1, 7, 2)  # Reshape back to 3D tensor (1, 7, 2)

# Predict the next 7 days
predicted_values = regressor.predict(inputs_scaled)

# Reshape the predictions to 2D for inverse transformation
predicted_values_flat = predicted_values.reshape(-1, 2)

# Apply the inverse transformation to get actual scale values
predicted_values_flat = sc.inverse_transform(predicted_values_flat)

# Get the actual values from dataset_test for the next 7 days
actual_values = dataset_test[['Weight', 'Calories intake']].iloc[:7].values

# Plotting actual vs predicted values for Weight and Calories in separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
# Import necessary libraries
import matplotlib.pyplot as plt

# Create two subplots: one for weight and one for calories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot actual and predicted weight on the first subplot (ax1)
ax1.plot(range(7), actual_values[:, 0], color='red', label='Actual Weight')  # Actual Weight
ax1.plot(range(7), predicted_values_flat[:, 0], color='blue', linestyle='--', label='Predicted Weight')  # Predicted Weight
ax1.set_title('Actual vs Predicted Weight for Next 7 Days')
ax1.set_xlabel('Days')
ax1.set_ylabel('Weight')
ax1.legend()
ax1.grid(True)

# Plot actual and predicted calories on the second subplot (ax2)
ax2.plot(range(7), actual_values[:, 1], color='green', label='Actual Calories')  # Actual Calories
ax2.plot(range(7), predicted_values_flat[:, 1], color='orange', linestyle='--', label='Predicted Calories')  # Predicted Calories
ax2.set_title('Actual vs Predicted Calories Intake for Next 7 Days')
ax2.set_xlabel('Days')
ax2.set_ylabel('Calories')
ax2.legend()
ax2.grid(True)

# Adjust layout for better spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
