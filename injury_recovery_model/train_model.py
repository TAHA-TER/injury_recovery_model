import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
df = pd.read_csv("injury1.csv")

# Define features and target variable
X = df.drop('Recovery_Period', axis=1)
y = df['Recovery_Period']

# Specify numerical and categorical features
numeric_features = ['Calorie', 'Age', 'Weight']
categorical_features = ['Gender', 'Type', 'Injury']

# Create a preprocessor for transforming the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained model to disk
joblib.dump(pipeline, 'injury_recovery_model.pkl')

print("Model trained and saved as 'injury_recovery_model.pkl'")
