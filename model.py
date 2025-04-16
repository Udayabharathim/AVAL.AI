import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Final_PCOS_Dataset_2000.csv", encoding='latin-1')
print(df['pcos'].value_counts())
# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "_", regex=True)
# Feature Engineering: Add interaction terms
df["bmi_hairloss_interaction"] = df["weight_in_kg"] / (df["height"] / 100) ** 2 * df["hair_loss"]
df["weight_period_duration"] = df["weight_in_kg"] * df["period_duration"]

# One-hot encode blood groups (DO NOT label encode)
df = pd.get_dummies(df, columns=["blood_group"], prefix="bg")

# Define features and target
features = [
    "age_in_years", "weight_in_kg", "height", "months_between_periods",
    "weight_gain_recently", "excess_body_facial_hair", "skin_darkening",
    "hair_loss", "acne", "fast_food", "exercise", "are_your_periods_regular_",
    "period_duration", "bmi_hairloss_interaction", "weight_period_duration"
] + [col for col in df.columns if "bg_" in col]  # Include one-hot blood groups

target = "pcos"
X = df[features]
y = df[target]

# Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale using training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

# Hyperparameter Tuning with GridSearchCV
# --- RandomForest ---
param_grid_rf = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy'
)
rf_grid.fit(X_train_bal, y_train_bal)

# --- XGBoost ---
param_grid_xgb = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42),
    param_grid_xgb,
    cv=5,
    scoring='accuracy'
)
xgb_grid.fit(X_train_bal, y_train_bal)

# --- CatBoost ---
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0,
    random_state=42
)
cat.fit(X_train_bal, y_train_bal)  # CatBoost handles imbalance well

# Build Stacking Classifier
base_models = [
    ('rf', rf_grid.best_estimator_),
    ('xgb', xgb_grid.best_estimator_),
    ('cat', cat)
]

stacker = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacker.fit(X_train_bal, y_train_bal)

# Cross-validated accuracy
cv_scores = cross_val_score(stacker, X_train_bal, y_train_bal, cv=5, scoring='accuracy')
print(f"\nCross-Validated Accuracy: {cv_scores.mean() * 100:.2f}% (±{cv_scores.std() * 100:.2f})")

# Final evaluation on test set
y_pred = stacker.predict(X_test_scaled)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Feature importance analysis
# explainer = shap.TreeExplainer(rf_grid.best_estimator_)
# shap_values = explainer.shap_values(X_train_bal)
# shap.summary_plot(shap_values, X_train_bal, feature_names=features, plot_type="bar")

# Save model
joblib.dump(stacker, "optimized_pcos_model.pkl")
print("\nOptimized model saved as optimized_pcos_model.pkl!")