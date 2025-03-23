import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Final_PCOS_Dataset.csv")
print("Dataset loaded successfully!")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_").str.replace(r"[^\w]", "", regex=True)
df.rename(columns={"have_you_been_diagnosed_with_pcospcod": "pcos", "height_in_cm__feet": "height"}, inplace=True)

# Compute BMI
df["bmi"] = df["weight_in_kg"] / (df["height"] / 100) ** 2

# Encode categorical variables
label_enc = LabelEncoder()
df["are_your_periods_regular_"] = label_enc.fit_transform(df["are_your_periods_regular_"])
df["blood_group"] = label_enc.fit_transform(df["blood_group"])

# Select features and target
features = [
    "age_in_years", "weight_in_kg", "height", "bmi", "blood_group", "months_between_periods", 
    "weight_gain_recently", "excess_body_facial_hair", "skin_darkening", "hair_loss", 
    "acne", "fast_food", "exercise", "are_your_periods_regular_", "period_duration"
]
target = "pcos"
X = df[features]
y = df[target]

# Handle missing values
imputer = SimpleImputer(strategy="mean")  # Use mean for numerical columns
X_imputed = imputer.fit_transform(X)

# Train-Test Split (before applying SMOTE to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Convert back to DataFrame
df_train_resampled = pd.DataFrame(X_train_resampled, columns=features)
df_train_resampled["pcos"] = y_train_resampled  # Add target column back

# Save the balanced dataset to CSV
output_path = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Balanced_PCOS_Dataset.csv"
df_train_resampled.to_csv(output_path, index=False)
print(f"\nBalanced dataset saved to: {output_path}")