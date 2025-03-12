import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Final_PCOS_Dataset.csv")
print("Dataset loaded successfully!")

# Standardize column names
print("Standardizing column names...")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace(r"[^\w]", "", regex=True)
print("Column names standardized!")

# Rename target column
print("Renaming target column...")
df.rename(columns={"have_you_been_diagnosed_with_pcospcod": "pcos"}, inplace=True)
print("Target column renamed!")

# Fix height column name
print("Fixing height column name...")
df.rename(columns={"height_in_cm__feet": "height"}, inplace=True)
print("Height column name fixed!")

# Compute BMI
print("Computing BMI...")
df["bmi"] = df["weight_in_kg"] / (df["height"] / 100) ** 2
print("BMI computed!")

# Encode categorical variables
print("Encoding categorical variables...")
label_enc = LabelEncoder()
df["are_your_periods_regular_"] = label_enc.fit_transform(df["are_your_periods_regular_"])
df["blood_group"] = label_enc.fit_transform(df["can_you_tell_us_your_blood_group"])
print("Categorical variables encoded!")

# Check for missing values
print("Checking for missing values...")
print("Missing values in the dataset:")
print(df.isnull().sum())
print("Missing values checked!")

# Check class distribution
print("\nChecking class distribution of target variable (PCOS)...")
print(df["pcos"].value_counts())
print("Class distribution checked!")

# Select features and target
print("Selecting features and target...")
features = ["age_in_years", "weight_in_kg", "height", "bmi", "blood_group", "months_between_periods", "weight_gain_recently", "excess_body_facial_hair", "skin_darkening", "hair_loss", "acne", "fast_food", "exercise", "mood_swings", "are_your_periods_regular_", "period_duration"]
target = "pcos"
X = df[features]
y = df[target]
print("Features and target selected!")

# Train-test split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets!")

# Hyperparameter tuning for RandomForestClassifier
print("\nStarting RandomForest hyperparameter tuning...")
param_grid_rf = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 6, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
print("RandomForest hyperparameter tuning completed!")

# Best RandomForest model
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nBest RandomForest Model Accuracy: {accuracy_rf * 100:.2f}%")

# Classification report for RandomForest
print("\nRandomForest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Hyperparameter tuning for XGBoost
print("\nStarting XGBoost hyperparameter tuning...")
param_grid_xgb = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0]
}

xgb = XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)
print("XGBoost hyperparameter tuning completed!")

# Best XGBoost model
best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\nBest XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%")

# Classification report for XGBoost
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Save the best model (RandomForest or XGBoost)
print("\nSaving the best model...")
if accuracy_rf > accuracy_xgb:
    best_model = best_rf
    print("RandomForest selected as the best model.")
else:
    best_model = best_xgb
    print("XGBoost selected as the best model.")

joblib.dump(best_model, "pcos_model.pkl")
print("Integrated model saved as pcos_model.pkl!")