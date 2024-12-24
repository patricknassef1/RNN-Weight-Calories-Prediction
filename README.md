Project Overview: Predicting Weight Using an RNN and Linear Regression 🤖📊

This project uses a combination of a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers and a Linear Regression model to predict the weight of a user for the next 7 days based on historical data. The RNN model processes the input data, handles missing values, scales the data, and makes predictions. The Linear Regression model is used as a secondary approach to compare and enhance the accuracy of the predictions, offering a more comprehensive analysis for personalized health and fitness tracking. 🏋️‍♂️

Detailed Steps 📝
Data Collection and Preprocessing: 📥

Loading the Dataset: The project starts by loading two CSV datasets: one for training (train_data.csv) and another for testing (test_data.csv). 📂
Handling Missing Values: Since real-world datasets often have missing values, we use SimpleImputer from sklearn to fill in the missing values in the Weight column using the mean of the available data. 🔄
Feature Scaling: 📈

To ensure that the neural network can learn effectively, we scale the Weight data using MinMaxScaler. This normalization step transforms the values into a range between 0 and 1, which helps prevent issues with gradient-based optimizers during training. ⚙️
Data Structuring for Time-Series: ⏳

Creating Time Windows: Since we want to predict the next 7 days based on the previous 7 days, the dataset is transformed into sequences of 7 time steps (representing the last week’s data) for the input (x_train) and the next 7 days' values for the target (y_train). 🔢
Reshaping Data: The data is reshaped into a 3D tensor, where the dimensions are [number of samples, time steps, features]. This is required for LSTM models, as they work with sequences of data. 🔄
Building the Models: 🏗️

RNN Model:

Sequential Model: We use the Sequential model from Keras to build the RNN. 🔧
Adding Layers: The model consists of four LSTM layers with 100 units each, followed by a Dropout layer to prevent overfitting. Dropout is set to 10% for each layer, meaning that 10% of the neurons are randomly dropped during training. ❌
Output Layer: The final layer is a Dense layer with 7 units (for 7 days) and 1 feature (Weight), followed by a Reshape to match the target shape. 🧩
Compile the Model: The model is compiled using the adam optimizer and the mean_squared_error loss function. ⚡
Linear Regression Model:

Model Fitting: After training the RNN model, a Linear Regression model is trained on the preprocessed data to predict the same target values (Weight). This serves as a baseline comparison and helps evaluate the performance of the RNN model. 📈
Prediction: The Linear Regression model generates predictions for the next 7 days' weight based on the same historical data used by the RNN model. 🔮
Training the Models: 🎓

The RNN model is trained on the preprocessed data for 30 epochs with a batch size of 15. Training is done on the input data (x_train) and target data (y_train). 🏃‍♂️
The Linear Regression model is trained on the same data and evaluated alongside the RNN model. ⚖️
Prediction: 🔮

Test Data Preparation: After training, both models make predictions for the next 7 days based on the test dataset. The models take the last 7 days of data from the combined training and test set, scale it, and predict the next 7 days' values. 📅
Inverse Scaling: Since both models output scaled values, these are transformed back to the original scale using the inverse of the MinMaxScaler. 🔄
Visualization: 📊

The actual vs predicted values for Weight are plotted separately for both models.
One plot for the predicted vs actual Weight 🏋️‍♂️
This helps evaluate the models' performance and compare how closely the predictions match the actual data from both the RNN and Linear Regression models. 🔍
