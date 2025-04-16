import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Final_PCOS_Dataset_4000.csv", encoding='latin-1')
print("Dataset loaded successfully!")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace(r"[^\w]", "", regex=True)

# Rename target column
df.rename(columns={"have_you_been_diagnosed_with_pcospcod": "pcos"}, inplace=True)

# Fix height column name
df.rename(columns={"height_in_cm__feet": "height"}, inplace=True)

# Compute BMI
df["bmi"] = df["weight_in_kg"] / (df["height"] / 100) ** 2

# Encode categorical variables
label_enc = LabelEncoder()
df["are_your_periods_regular_"] = label_enc.fit_transform(df["are_your_periods_regular_"])
df["blood_group"] = label_enc.fit_transform(df["blood_group"])

# Check for missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Select features and target
features = [
    "age_in_years", "weight_in_kg", "height", "bmi", "blood_group", "months_between_periods", 
    "weight_gain_recently", "excess_body_facial_hair", "skin_darkening", "hair_loss", 
    "acne", "fast_food", "exercise", "are_your_periods_regular_", "period_duration"
]
target = "pcos"
X = df[features]
y = df[target]

# Apply feature scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rf.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# Train CatBoost
cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
cat.fit(X_train, y_train)

# Voting Classifier (Ensemble Model)
voting_clf = VotingClassifier(estimators=[
    ('rf', rf), ('xgb', xgb), ('cat', cat)
], voting='soft')  # 'soft' uses probabilities for better performance
voting_clf.fit(X_train, y_train)

# Make predictions
y_pred = voting_clf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nEnsemble Model Classification Report:")
print(classification_report(y_test, y_pred))

# Print accuracy
print(f"\nFinal Ensemble Model Accuracy: {accuracy * 100:.2f}%")

# Save the ensemble model
joblib.dump(voting_clf, "pcos_ensemble_model.pkl")
print("Best Ensemble model saved as pcos_ensemble_model.pkl!")
