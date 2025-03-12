import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "data\Final_PCOS_Dataset_2000.csv"  # Update path if needed
df = pd.read_csv(file_path)

# Print column names to debug missing columns
print("Columns in Dataset:", df.columns.tolist())

# Ensure the expected column exists
expected_col = "How long does your period last ?"
matching_cols = [col for col in df.columns if "period" in col.lower()]
if not any(expected_col in col for col in matching_cols):
    print(f"⚠️ Warning: '{expected_col}' column not found. Did you mean one of these? {matching_cols}")

# Convert categorical columns to numerical (Ensure they exist)
categorical_cols = [
    "Do you have excessive body/facial hair growth ?", 
    "Are you noticing skin darkening recently?",
    "Do have hair loss/hair thinning/baldness ?", 
    "Do you have pimples/acne on your face/jawline ?",
    "Do you eat fast food regularly ?", 
    "Do you exercise on a regular basis ?",
    "Do you experience mood swings ?", 
    "Are your periods regular ?"
]

# Ensure categorical columns exist before encoding
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
label_encoder = LabelEncoder()
for col in existing_categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))  # Convert to string to handle NaNs

# Normalize numerical columns (Ensure they exist)
numerical_cols = ["Age (in Years)", "Weight (in Kg)", "Height (in Cm / Feet)", expected_col]
existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

scaler = StandardScaler()
df[existing_numerical_cols] = scaler.fit_transform(df[existing_numerical_cols])

# Separate Features (X) and Target (y)
target_col = "Have you been diagnosed with PCOS/PCOD?"
if target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    print(f"⚠️ Warning: Target column '{target_col}' not found!")
    X, y = None, None

# Split Data into Train & Test Sets (Only if target column exists)
if X is not None and y is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data Preprocessing Completed!")
else:
    print("❌ Error: Data preprocessing failed due to missing columns.")

