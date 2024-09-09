
# Overview
This project aims to predict future Google stock prices using Long Short-Term Memory (LSTM) networks.

# Libraries Used
NumPy: Numerical computing library.

Pandas: Data analysis and manipulation library.

Matplotlib: Plotting library.

Keras: Deep learning library built on top of TensorFlow.

# Data
The project utilizes two CSV files:

**Google_Stock_Price_Train.csv:**
Contains historical Google stock price data used for training the model.

**Google_Stock_Price_Test.csv:**
Contains data for which we want to predict future stock prices.

# Preprocessing

Data Loading: The code reads both CSV files using pandas read_csv function.

Feature Selection: It selects the "Open" price column from both datasets.

Scaling: The code uses MinMaxScaler from scikit-learn to scale the data between 0 and 1 for better model performance with neural networks.

# Model Building (LSTM)

## Data Preparation:
The code creates sequences of past 60 days' opening prices for training.

It separates these sequences as features (X_train) and the next day's opening price as target variable (y_train).
## Model Definition: 
A sequential LSTM model is built using Keras:

Three LSTM layers with 50 units each are used with dropout regularization to prevent overfitting.

A final Dense layer with one unit predicts the next day's opening price.
## Model Compilation: 
The model is compiled with the Adam optimizer and mean squared error (MSE) loss function.
## Model Training: 
The model is trained on the prepared training data (X_train, y_train) for 100 epochs with a batch size of 32.
# Prediction
## Test Data Preparation: 
Similar to training data, sequences of the last 60 days' opening prices are created from the test dataset.
## Prediction Generation: 
The trained model predicts the next day's opening price for each sequence in the test data.
## Inverse Scaling: 
The predicted values are inverse-transformed back to the original scale using the MinMaxScaler.
# Evaluation
## Plotting: 
The code plots both actual and predicted stock prices for the test data to visually compare the model's performance.

# Additional Notes
You may need to adjust the number of LSTM layers, units, and training epochs based on your specific dataset and requirements.

Consider using other evaluation metrics like mean absolute error (MAE) or root mean squared error (RMSE) to assess the model's performance.

Explore techniques like hyperparameter tuning to optimize the model's parameters.

For production use, consider deploying the model as a web API or integrating it into a larger application.
