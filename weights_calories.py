# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression

# Data Preprocessing
dataset_train = pd.read_csv("D:\\project final year\\train_data.csv")

# Filling missing values with mean using imputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset_train[['Weight', 'Calories intake']] = imputer.fit_transform(
    dataset_train[['Weight', 'Calories intake']]
)

# Scaling the dataset
sc = MinMaxScaler(feature_range=(0, 1))
scaled_training_set = sc.fit_transform(dataset_train[['Weight', 'Calories intake']])  # Scaling only relevant columns

# Creating input and output for training
x_train = []
y_train = []
time_steps = 7

# Process data for a sliding window of 7 days
for i in range(time_steps, len(scaled_training_set) - time_steps):
    x_train.append(scaled_training_set[i - time_steps:i, :])  # Last 7 days of all features
    y_train.append(scaled_training_set[i:i + time_steps, 0])  # Next 7 days of weight (column 0)

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshaping the data to match LSTM input (7 days, 2 features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))

    # Building the Recurrent Neural Network (RNN) - LSTM Model
    regressor = Sequential()
    
    # First LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 2)))
    regressor.add(Dropout(0.2))
    
    # Second LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    # Third LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=True))
    regressor.add(Dropout(0.2))
    
    # Final LSTM layer with dropout
    regressor.add(LSTM(units=100, return_sequences=False))
    regressor.add(Dropout(0.2))
    
    # Output layer (predicting 7 days, 1 feature - weight)
    regressor.add(Dense(units=7))  # Predicting 7 time steps for weight
    
    # Compile the model with Adam optimizer and MSE loss
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    
    # Early stopping callback
    early_stopping_loss = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Fit the model with early stopping
    regressor.fit(
        x_train, 
        y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.1, 
        callbacks=[early_stopping_loss]
    )

# Extracting features from the LSTM output for Linear Regression
lstm_output = regressor.predict(x_train)  # This will give 7 days of weight predictions (for each input)

# Flatten the LSTM output to use as input to Linear Regression
lstm_output_flat = lstm_output.reshape(lstm_output.shape[0], -1)  # Flatten to (samples, 7)

# Train the Linear Regression model on the LSTM output
linear_regressor = LinearRegression()
linear_regressor.fit(lstm_output_flat, y_train)

# Now, you can use the Linear Regression model to predict the weight
# For the prediction, we'll also use the LSTM's output for the new input data

# Load the test dataset
dataset_test = pd.read_csv("D:\\project final year\\result.csv")

# Load the input data for prediction
file_path = 'D:\\project final year\\inputs.csv'
inputs = pd.read_csv(file_path)

# Ensure the same columns are used during prediction (Weight and Calories intake)
inputs_scaled = sc.transform(inputs[['Weight', 'Calories intake']])

# Reshape the input data to the expected shape (1, 7, 2)
inputs_scaled = inputs_scaled.reshape(1, 7, 2)

# Get the LSTM output for the input data
lstm_output_for_input = regressor.predict(inputs_scaled)

# Flatten the LSTM output and make predictions using Linear Regression
lstm_output_for_input_flat = lstm_output_for_input.reshape(1, -1)  # Flatten to (1, 7)
predicted_weight = linear_regressor.predict(lstm_output_for_input_flat)
# Make sure the predicted_weight is reshaped to 2D before applying inverse transformation
# predicted_weight originally has shape (7, 1), we need to combine it with zeros for the Calories intake feature
predicted_weight_reshaped = np.reshape(predicted_weight, (-1, 1))  # Ensure it's 2D with shape (7, 1)

# Add zeros for the Calories intake feature (second column)
predicted_values_with_zeros = np.hstack((predicted_weight_reshaped, np.zeros_like(predicted_weight_reshaped)))

# Apply inverse transformation for the combined data (Weight and Calories intake)
predicted_weight = sc.inverse_transform(predicted_values_with_zeros)[:, 0]

# Get the actual values for the next 7 days from the test dataset
actual_values = dataset_test[['Weight']].values

# Plotting actual vs predicted values for Weight
plt.figure(figsize=(10, 6))

# Plot actual and predicted weight
plt.plot(range(7), actual_values[:, 0], color='red', label='Actual Weight')  # Actual Weight
plt.plot(range(7), predicted_weight, color='blue', linestyle='--', label='Predicted Weight')  # Predicted Weight

# Add plot titles and labels
plt.title('Actual vs Predicted Weight for Next 7 Days')
plt.xlabel('Days')
plt.ylabel('Weight')
plt.ylim(50, 90)  # Adjust y-axis range based on your data
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
