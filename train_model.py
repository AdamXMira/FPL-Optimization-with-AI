import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np

# Load cleaned data
df = pd.read_csv("cleaned_player_data.csv")

# Encode season as a number
# This is useful for model training and evaluation
season_map = {
    '2021-2022': 1,
    '2022-2023': 2,
    '2023-2024': 3,
    '2024-2025': 4
}
df['season_num'] = df['season'].map(season_map)

# Define features and target variable
# Features: all columns except 'full_name', 'season', 'total_points', 'now_cost', 'selected_by_percent'
# Target: 'total_points'
feature_cols = [
    col for col in df.columns

    if col not in ['full_name', 'season', 'total_points', 'now_cost', 'selected_by_percent']
]
X = df[feature_cols]
y = df['total_points']

# Sample weights (more recent seasons = higher weight)
# This is to ensure the model focuses more on recent seasons
# and less on older seasons
# The weights initially included the 2021-2022, 2022-2023 seasons, but it was removed
sample_weights = df['season'].map({
    '2023-2024': 1.5,
    '2024-2025': 2.0
})

# === Train-test split ===
# Stratified split based on 'season' to ensure each season is represented in both train and test sets
# This is to ensure the model is trained on a representative sample of the data
# and to avoid overfitting to a specific season
# The random_state is set to 42 for reproducibility
# The test size is set to 20% of the data
X_train, X_test, y_train, y_test, train_weights, test_meta = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, stratify=df['season']
)

# Define hyperparameter search space
# using RandomizedSearchCV for hyperparameter tuning

# - n_estimators: number of trees
# - max_depth: tree depth (to manage model complexity)
# - learning_rate: step size shrinkage
# - subsample: fraction of training instances per tree
# - colsample_bytree: fraction of features per tree
# - gamma: minimum loss function for a single split
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'gamma': uniform(0, 5)
}

# Base XGBoost model
# Using 'reg:squarederror' for regression tasks
# The random_state is set to 42 for reproducibility
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Randomized search over hyperparameters
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,                      # Number of parameter combinations to try
    scoring='neg_mean_squared_error',  # Evaluation metric (negative since scipy thinks higher is better)
                                       #Returns the combination that yields the lowest MSE
    cv=3,                           # 3-fold cross-validation
    verbose=1,                      # Print progress to console
    random_state=42,
    n_jobs=-1                       # Use all available cores
)

# Fit search using weighted training data 
search.fit(X_train, y_train, sample_weight=train_weights)

# Get the best model found
model = search.best_estimator_ #Based on the best hyperparameters from the search
#print("Best hyperparameters:", search.best_params_)

# Model Evaluation and prediction
# Predict on the test set
y_pred = model.predict(X_test)
# The model is evaluated using the test set
# The mean squared error (MSE) is calculated to assess the model's performance
# The MSE is a measure of the average squared difference between predicted and actual values
# A lower MSE indicates better model performance
# The R² score is also calculated to assess the model's performance
# The R² score is a measure of how well the model explains the variance in the data
# A higher R² score indicates better model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"Test R² Score: {r2:.2f}")

# Final model-wide predictions
df['predicted_points'] = model.predict(X)

# Keep only 2024-2025 predictions for later use in the optimization module
# === Filter for current season only ===
current_season_df = df[df['season'] == '2024-2025'].copy()

# === Round now_cost to nearest 0.5M ===
# now_cost is in tenths of millions (e.g., 105 = £10.5M), so:
# Divide by 10 → round to nearest 0.5 → multiply by 10
current_season_df['now_cost'] = (current_season_df['now_cost'] / 5).round() * 5
current_season_df['now_cost'] = current_season_df['now_cost'].astype(int)

# === Round selected_by_percent (ownership) to nearest 0.5 ===
current_season_df['ownership'] = (current_season_df['selected_by_percent'])

# === Sort by predicted points ===
current_season_df = current_season_df.sort_values(by='predicted_points', ascending=False)

# === Save to CSV ===
current_season_df[[
    'full_name',
    'predicted_points',
    'now_cost',
    'ownership',
    'total_points',
    'minutes', 'position'
]].to_csv("player_predictions.csv", index=False)

print("✅ Saved cleaned and rounded predictions to player_predictions.csv")

# Plot predicted vs actual points

# The error is the absolute difference between the actual and predicted values
# This is to visualize the model's performance and improve graph readability
error = np.abs(y_test - y_pred)

plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred, hue=error, palette="coolwarm", legend=False)

# Plot ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--', color='black', linewidth=1.5, label='Perfect Prediction Line'
)

# Axis labels & title 
# The x-axis is the actual total points
# The y-axis is the predicted total points
plt.xlabel("Actual Total Points", fontsize=12)
plt.ylabel("Predicted Total Points", fontsize=12)
plt.title("XGBoost Predicted vs. Actual Total Points", fontsize=14)


# Get indices of top 3 absolute errors (largest prediction gaps)
top_errors_idx = np.argsort(-error)[:3]

# Loop through each of those high-error cases to label them 
# on the plot
for idx in top_errors_idx:
    actual = y_test.iloc[idx]             # Actual total points
    predicted = y_pred[idx]               # Model-predicted points

    # Get the player's full name from the original DataFrame
    label = f"{df.iloc[y_test.index[idx]]['full_name']}"

    # Annotate the point slightly offset to avoid overlap
    plt.text(actual + 1, predicted + 1, label, fontsize=9, color='gray')

# Add legend and save the plot
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=300)
plt.close()







