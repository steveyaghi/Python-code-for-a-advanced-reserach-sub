# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers


#%%
data = np.load('microMacroData.npy')
print(f'Dimension of this dataset: {data.shape}')

# Macro parameter names
macro_param_names = [r'$G_0$ (MPa)', r'$M_{tc}$', r'$N$', r'$\Gamma$', r'$\lambda$', r'$\chi$', r'$H_0$ ($10^2$)', r'$H_a$ ($10^2$)']

# Micro parameter names
micro_param_names = [r'$E_c$ (GPa)', r'$k_r$', r'$\mu_r$', r'$\mu_s$']

# Combine macro and micro parameter names
column_names = macro_param_names + micro_param_names

# Create a DataFrame with column names
df = pd.DataFrame(data, columns=column_names)

# Print the DataFrame with column names
print(df)

list_of_macro_params = []
list_of_micro_params = []
for i in range(data.shape[0]):
    macro_params, micro_params = data[i, :8], data[i, 8:]
    list_of_macro_params.append(macro_params)
    list_of_micro_params.append(micro_params)
#%%
################################################ Histogram plotting
# Create a DataFrame with column names
df = pd.DataFrame(data, columns=column_names)

# Print the DataFrame with column names
print(df)

# Get the number of columns in the dataset
num_columns = data.shape[1]

# Set up subplots for each histogram
fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(8, 2 * num_columns))

# Plot histograms for each column in the same figure
for i in range(num_columns):
    axes[i].hist(data[:, i], bins=20, color='blue', alpha=0.7)
    axes[i].set_title(f'Histogram of {column_names[i]}')
    axes[i].set_xlabel('Values')
    axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()
#%%    
############################################## scatter plotting
macro_data = data[:, :len(macro_param_names)]
micro_data = data[:, len(macro_param_names):]

# Get the number of macro and micro variables
num_macro_variables = macro_data.shape[1]
num_micro_variables = micro_data.shape[1]

# Set up a grid of scatter plots
fig, axes = plt.subplots(num_macro_variables, num_micro_variables, figsize=(15, 12))

# Loop through each combination of macro and micro variables
for i in range(num_macro_variables):
    for j in range(num_micro_variables):
        # Scatter plot for the i-th macro variable against the j-th micro variable
        axes[i, j].scatter(macro_data[:, i], micro_data[:, j], alpha=0.5)
        axes[i, j].set_xlabel(macro_param_names[i])
        axes[i, j].set_ylabel(micro_param_names[j])
        axes[i, j].set_title(f'Scatter: {macro_param_names[i]} vs {micro_param_names[j]}')
        
  # Calculate R-squared value
        X = macro_data[:, i].reshape(-1, 1)
        y = micro_data[:, j]
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)
        axes[i, j].text(0.5, 0.9, f'R-squared: {r_squared:.2f}', transform=axes[i, j].transAxes, ha='center', va='center')



plt.tight_layout()
plt.show()
#%%
################################################ PCA
# Assuming 'data' is your dataset with macro and micro parameters
macro_params = data[:, :8]
micro_params = data[:, 8:]

# Standardize the data
standardized_macro_params = (macro_params - np.mean(macro_params, axis=0)) / np.std(macro_params, axis=0)
standardized_micro_params = (micro_params - np.mean(micro_params, axis=0)) / np.std(micro_params, axis=0)

# Apply PCA to macro parameters
pca_macro = PCA(n_components=6)  # Choose the number of components based on your analysis
transformed_macro_params = pca_macro.fit_transform(standardized_macro_params)

# Apply PCA to micro parameters
pca_micro = PCA(n_components=4)  # Choose the number of components based on your analysis
transformed_micro_params = pca_micro.fit_transform(standardized_micro_params)

# Plot explained variance for macro parameters
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, pca_macro.n_components_ + 1), pca_macro.explained_variance_ratio_)
plt.xlabel('Principal Components (Macro)')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance for Macro Parameters')

# Plot explained variance for micro parameters
plt.subplot(1, 2, 2)
plt.bar(range(1, pca_micro.n_components_ + 1), pca_micro.explained_variance_ratio_)
plt.xlabel('Principal Components (Micro)')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance for Micro Parameters')

plt.tight_layout()
plt.show()


############################################### PCA
# =============================================================================
# 
#Standardize the data (important for PCA)
standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

 # Apply PCA
pca = PCA()
principal_components = pca.fit_transform(standardized_data)


# Get loadings for the first 6 principal components
loadings = pca.components_[:6, :]

# Print or visualize the loadings
print("Loadings for the first 6 principal components:")
print(loadings)

feature_names = macro_param_names + micro_param_names
loadings_df = pd.DataFrame(loadings, columns=feature_names, index=[f'PC{i+1}' for i in range(6)])

print("Loadings for the first 6 principal components:")
print(loadings_df)
####################################################
#%%
# Apply PCA to macro parameters
pca_macro = PCA(n_components=4)  # Choose the number of components based on your analysis
transformed_macro_params = pca_macro.fit_transform(standardized_data[:, :8])

# Apply PCA to micro parameters
pca_micro = PCA(n_components=4)  # Choose the number of components based on your analysis
transformed_micro_params = pca_micro.fit_transform(standardized_data[:, 8:])

# Define figure size
fig, axes = plt.subplots(4, 4, figsize=(16, 12))

# Plot macro PCs vs micro variables
for i in range(pca_macro.n_components_):
    for j in range(pca_micro.n_components_):
        i_row = int(i / 4)
        i_col = i % 4
        j_row = int(j / 4)
        j_col = j % 4

        # Check if the indices are within the valid range
        if i < pca_macro.n_components_ and j < pca_micro.n_components_:
        
            # Create subplot and plot data
            ax = plt.subplot(4, 4, i * 4 + j + 1)
            plt.scatter(transformed_macro_params[:, i], transformed_micro_params[:, j], alpha=0.5)
            plt.xlabel(f'PC{i + 1} (Macro)')
            plt.ylabel(f'PC{i + 1} (Micro)')
      
 # Calculate R-squared value
            X = transformed_macro_params[:, i].reshape(-1, 1)
            y = transformed_micro_params[:, j]
            reg = LinearRegression().fit(X, y)
            r_squared = reg.score(X, y)
            plt.text(0.5, 0.9, f'R-squared: {r_squared:.2f}', transform=ax.transAxes, ha='center', va='center')

# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()


#################################################### polynomial regression 
#%%
# Apply PCA to macro parameters
pca_macro = PCA(n_components=4)  # Choose the number of components based on your analysis
transformed_macro_params = pca_macro.fit_transform(standardized_macro_params)

# Extract the first 3 PCA components
macro_pca_components = transformed_macro_params[:, :4]

# Concatenate the first 3 PCA components with micro parameters
X = np.concatenate((macro_pca_components, standardized_micro_params), axis=1)

# Assuming you want to predict the first micro parameter, change the index accordingly
y = standardized_micro_params[:, 0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Regression
degree = 3  # Choose the degree of the polynomial
poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Fit the Polynomial Regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on the test set
y_pred = model.predict(X_poly_test)

# Calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Create a grid of subplots for each micro variable
num_micro_variables = standardized_micro_params.shape[1]
fig, axes = plt.subplots(nrows=num_micro_variables, ncols=4, figsize=(15, 4 * num_micro_variables))

for i in range(num_micro_variables):
    y = standardized_micro_params[:, i]

    for j in range(4):  # Iterate over the first 3 PCA components
        X = transformed_macro_params[:, j].reshape(-1, 1)

        # Apply Polynomial Regression
        degree = 2  # Choose the degree of the polynomial
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        # Fit the Polynomial Regression model
        model = LinearRegression()
        model.fit(X_poly, y)

        # Predict on the same X values for a smooth curve
        X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_pred_poly = poly.transform(X_pred)
        y_pred = model.predict(X_pred_poly)

        # Compute R-squared value
        r2 = r2_score(y, model.predict(X_poly))

        # Plot the results
        axes[j, i].scatter(X, y, color='black', label='Actual')
        axes[j, i].plot(X_pred, y_pred, color='red', label=f'Predicted (R-squared: {r2:.2f})')
        axes[j, i].set_xlabel(f'PCA Component {j + 1}')
        axes[j, i].set_ylabel(f'Micro Variable {i + 1}')
        axes[j, i].legend()

plt.tight_layout()
plt.show()



######################################################### Neural Network code:
 #%%   
# Standardize the features
scaler = StandardScaler()
macro_params_scaled = scaler.fit_transform(data[:, :8])
micro_params_scaled = scaler.fit_transform(data[:, 8:])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(macro_params_scaled, micro_params_scaled, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(4)  # Output layer for regression, assuming there are 4 micro parameters
])

# Specify the learning rate for the Adam optimizer
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # You can adjust the learning rate (e.g., 0.001)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Visualize training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize predictions vs actual values for each micro parameter
num_micro_params = y_test.shape[1]
fig, axes = plt.subplots(nrows=num_micro_params, ncols=1, figsize=(8, 4 * num_micro_params))

for i in range(num_micro_params):
    # Scatter plot of actual vs predicted values
    axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    axes[i].set_xlabel(f'Actual Micro Parameter {i+1}')
    axes[i].set_ylabel(f'Predicted Micro Parameter {i+1}')
    axes[i].set_title(f' (3 layers Neural network) Micro Parameter {i+1}: Actual vs Predicted')

    # Calculate and display R-squared value
    r2_value = r2_score(y_test[:, i], y_pred[:, i])
    axes[i].text(0.1, 0.9, f'R-squared: {r2_value:.2f}', transform=axes[i].transAxes, ha='left', va='center')

plt.tight_layout()
plt.show()

# Calculate and print R-squared value for each micro parameter
r2_values = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_micro_params)]
for i, r2_value in enumerate(r2_values):
    print(f'R-squared value for Micro Parameter (before HP optimization) {i+1}: {r2_value}')

################################################################## Random search function
#%%
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score

# Define the function to create the neural network model
def create_model(learning_rate=0.001, units_layer1=64, units_layer2=32, num_output_units=4):
    model = Sequential([
        layers.Dense(units_layer1, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(units_layer2, activation='relu'),
        layers.Dense(num_output_units)  # Output layer for all micro parameters
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


# Example: For 4 output variables
keras_regressor = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=0, num_output_units=4)


# Define hyperparameter search space
param_dist = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'units_layer1': [32, 64, 128],
    'units_layer2': [16, 32, 64]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=keras_regressor,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=3,       # Number of cross-validation folds
    verbose=2,
    n_jobs=-1,    # Use all available processors
    scoring=make_scorer(r2_score, greater_is_better=True)  # Specify the scoring metric
)

# Perform the random search
random_search.fit(X_train, y_train)  # Train to predict all micro parameters

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Calculate and print R-squared value for all micro parameters
r2_value_best = r2_score(y_test, y_pred_best)
print(f'R-squared value for all Micro Parameters (after optimization): {r2_value_best}')


################################################################### Neural network with more layers 
#%%
# Standardize the features
scaler = StandardScaler()
macro_params_scaled = scaler.fit_transform(data[:, :8])
micro_params_scaled = scaler.fit_transform(data[:, 8:])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(macro_params_scaled, micro_params_scaled, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(4)  # Output layer for regression, assuming there are 4 micro parameters
])

# Specify the learning rate for the Adam optimizer
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # You can adjust the learning rate (e.g., 0.001)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=2)

# Visualize training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize predictions vs actual values for each micro parameter
num_micro_params = y_test.shape[1]
fig, axes = plt.subplots(nrows=num_micro_params, ncols=1, figsize=(8, 4 * num_micro_params))

for i in range(num_micro_params):
    # Scatter plot of actual vs predicted values
    axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    axes[i].set_xlabel(f'Actual Micro Parameter  {i+1}')
    axes[i].set_ylabel(f'Predicted Micro Parameter  {i+1}')
    axes[i].set_title(f'(6 layers)Micro Parameter {i+1}: Actual vs Predicted (after randomsearch)')

    # Calculate and display R-squared value
    r2_value = r2_score(y_test[:, i], y_pred[:, i])
    axes[i].text(0.1, 0.9, f'R-squared: {r2_value:.2f}', transform=axes[i].transAxes, ha='left', va='center')

plt.tight_layout()
plt.show()

# Calculate and print R-squared value for each micro parameter
r2_values = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(num_micro_params)]
for i, r2_value in enumerate(r2_values):
    print(f'R-squared value for Micro Parameter (before HP optimization) {i+1}: {r2_value}')
################################################################################ neural network with PCA and micro
#%%
# Apply PCA to macro parameters
pca_macro = PCA(n_components=4)
transformed_pca_params = pca_macro.fit_transform(macro_params)

# Standardize micro parameters
scaler_micro = StandardScaler()
standardized_micro_params = scaler_micro.fit_transform(micro_params)

# Split the data into training and testing sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    transformed_pca_params, standardized_micro_params, test_size=0.2, random_state=42
)

# Build a neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(transformed_pca_params.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(standardized_micro_params.shape[1])  # Output layer for regression
])

# Specify the learning rate for the Adam optimizer
custom_optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train_pca, y_train_pca, epochs=100, validation_split=0.2, verbose=2)

# Visualize training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred_pca = model.predict(X_test_pca)

# Visualize predictions vs actual values for each micro parameter
num_micro_params = y_test_pca.shape[1]
fig, axes = plt.subplots(nrows=num_micro_params, ncols=1, figsize=(8, 4 * num_micro_params))

for i in range(num_micro_params):
    # Scatter plot of actual vs predicted values
    axes[i].scatter(y_test_pca[:, i], y_pred_pca[:, i], alpha=0.5)
    axes[i].set_xlabel(f'Actual Micro Parameter {i+1}')
    axes[i].set_ylabel(f'Predicted Micro Parameter {i+1}')
    axes[i].set_title(f' (PCA neural network before random search) Micro Parameter {i+1}: Actual vs Predicted using PCAs ')

    # Calculate and display R-squared value
    r2_value = r2_score(y_test_pca[:, i], y_pred_pca[:, i])
    axes[i].text(0.1, 0.9, f'R-squared: {r2_value:.2f}', transform=axes[i].transAxes, ha='left', va='center')

plt.tight_layout()
plt.show()
###################################################################### using randomsearch 
#%%
from tensorflow.keras.layers import Dense

# Define the function to create the neural network model
def create_model(learning_rate=0.001, units_layer1=64, units_layer2=32):
    model = Sequential([
        Dense(units_layer1, activation='relu', input_shape=(X_train_pca.shape[1],)),
        Dense(units_layer2, activation='relu'),
        Dense(y_train_pca.shape[1])  # Output layer for all micro parameters
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Example: For 4 output variables
keras_regressor = KerasRegressor(build_fn=create_model, epochs=100, batch_size=32, verbose=0)

# Define hyperparameter search space
param_dist = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'units_layer1': [32, 64, 128],
    'units_layer2': [16, 32, 64]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=keras_regressor,
    param_distributions=param_dist,
    n_iter=10,  # Number of random combinations to try
    cv=3,       # Number of cross-validation folds
    verbose=2,
    n_jobs=-1,    # Use all available processors
    scoring=make_scorer(r2_score, greater_is_better=True)  # Specify the scoring metric
)

# Perform the random search
random_search.fit(X_train_pca, y_train_pca)  # Train to predict all micro parameters

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate the best model on the test set
best_model_pca = random_search.best_estimator_
y_pred_pca_best = best_model_pca.predict(X_test_pca)

# Calculate and print R-squared value for all micro parameters
r2_value_pca_best = r2_score(y_test_pca, y_pred_pca_best)
print(f'R-squared value for all Micro Parameters (after optimization): {r2_value_pca_best}')

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)
########################################################################## PCA neurak network after random search
#%%
# Build a neural network model
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(transformed_pca_params.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(standardized_micro_params.shape[1])  # Output layer for regression
])

# Specify the learning rate for the Adam optimizer
custom_optimizer = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(X_train_pca, y_train_pca, epochs=100, validation_split=0.2, verbose=2)

# Visualize training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions on the test set
y_pred_pca = model.predict(X_test_pca)

# Visualize predictions vs actual values for each micro parameter
num_micro_params = y_test_pca.shape[1]
fig, axes = plt.subplots(nrows=num_micro_params, ncols=1, figsize=(8, 4 * num_micro_params))

for i in range(num_micro_params):
    # Scatter plot of actual vs predicted values
    axes[i].scatter(y_test_pca[:, i], y_pred_pca[:, i], alpha=0.5)
    axes[i].set_xlabel(f'Actual Micro Parameter {i+1}')
    axes[i].set_ylabel(f'Predicted Micro Parameter {i+1}')
    axes[i].set_title(f'(PCA Neural network after random search) Micro Parameter {i+1}: Actual vs Predicted using PCAs ')

    # Calculate and display R-squared value
    r2_value = r2_score(y_test_pca[:, i], y_pred_pca[:, i])
    axes[i].text(0.1, 0.9, f'R-squared: {r2_value:.2f}', transform=axes[i].transAxes, ha='left', va='center')

plt.tight_layout()
plt.show()


#############################
#%%


# Define figure size
num_macro_pcs = pca_macro.n_components_
num_micro_vars = standardized_micro_params.shape[1]  # Number of micro variables

fig, axes = plt.subplots(num_macro_pcs, num_micro_vars, figsize=(4 * num_micro_vars, 4 * num_macro_pcs))

# Plot macro PCs vs micro variables
for i in range(num_macro_pcs):
    for j in range(num_micro_vars):
        # Create subplot and plot data
        ax = plt.subplot(num_macro_pcs, num_micro_vars, i * num_micro_vars + j + 1)
        plt.scatter(transformed_macro_params[:, i], standardized_data[:, 8 + j], alpha=0.5)  # Adjust index for micro variables
        plt.xlabel(f'PC{i + 1} (Macro)')
        plt.ylabel(micro_param_names[j])  # Use the appropriate micro variable name

        # Calculate R-squared value
        X = transformed_macro_params[:, i].reshape(-1, 1)
        y = standardized_data[:, 8 + j]  # Adjust index for micro variables
        reg = LinearRegression().fit(X, y)
        r_squared = reg.score(X, y)
        plt.text(0.5, 0.9, f'R-squared: {r_squared:.2f}', transform=ax.transAxes, ha='center', va='center')

# Rotate x-labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()


################################################
#%%

# Apply PCA to macro parameters
pca_macro = PCA(n_components=4)  # Choose the number of components based on your analysis
transformed_macro_params = pca_macro.fit_transform(standardized_macro_params)

# Create a grid of subplots for each micro variable
num_micro_variables = standardized_micro_params.shape[1]
fig, axes = plt.subplots(nrows=num_micro_variables, ncols=4, figsize=(15, 4 * num_micro_variables))

for i in range(num_micro_variables):
    y = standardized_micro_params[:, i]

    for j in range(3):  # Iterate over the first 3 PCA components of macro parameters
        X = transformed_macro_params[:, j].reshape(-1, 1)

        # Apply Polynomial Regression
        degrees = [2, 3, 4]  # Choose the degrees of the polynomials to visualize
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            # Fit the Polynomial Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Calculate Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)

            # Plot the results
            axes[i, j].scatter(X, y, color='black', label='Actual')
            X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            X_pred_poly = poly.transform(X_pred)
            y_pred = model.predict(X_pred_poly)
            axes[i, j].plot(X_pred, y_pred, label=f'Degree {degree} (MSE: {mse:.2f})')

    axes[i, j].set_xlabel(f'PCA Component {j + 1}')
    axes[i, j].set_ylabel(f'Micro Variable {i + 1}')
    axes[i, j].legend()

plt.tight_layout()
plt.show()

############################## ############################
#%%

# Apply PCA to macro parameters
pca_macro = PCA(n_components=3)  # Choose the number of components based on your analysis
transformed_macro_params = pca_macro.fit_transform(standardized_macro_params)

# Extract the first 3 PCA components
macro_pca_components = transformed_macro_params[:, :3]

# Create a grid of subplots for each micro variable
num_micro_variables = standardized_micro_params.shape[1]
fig, axes = plt.subplots(nrows=num_micro_variables, ncols=3, figsize=(15, 4 * num_micro_variables))

# Loop through each micro variable
for i in range(num_micro_variables):
    y = standardized_micro_params[:, i]

    # Concatenate the first 3 PCA components with micro parameters
    X = np.concatenate((macro_pca_components, standardized_micro_params), axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the Mean Squared Error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(f'Micro Variable {i + 1} - Mean Squared Error: {mse:.2f}, R-squared: {r_squared:.2f}')

    # Plot the results
    for j in range(3):  # Iterate over the first 3 PCA components
        X_plot = transformed_macro_params[:, j]
        y_pred_combined = model.predict(np.column_stack((X_plot, standardized_micro_params)))

        # Compute R-squared value
        r2_combined = r2_score(y, y_pred_combined)

        # Plot the results
        axes[i, j].scatter(X_plot, y, color='black', label='Actual')
        axes[i, j].scatter(X_plot, y_pred_combined, color='red', label=f'Predicted (R-squared: {r2_combined:.2f})')
        axes[i, j].set_xlabel(f'PCA Component {j + 1}')
        axes[i, j].set_ylabel(f'Micro Variable {i + 1}')
        axes[i, j].legend()

plt.tight_layout()
plt.show()

