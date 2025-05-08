import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
# Load the dataset
df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\PCOS_comparision\AVAL.AI\data\PCOS_data.csv")

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Select relevant columns
selected_columns = [
    'PCOS (Y/N)', 'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI',
    'Blood Group', 'Cycle(R/I)', 'Cycle length(days)', 'Weight gain(Y/N)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)',
    'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)'
]
df_selected = df[selected_columns].copy()

# Drop rows with missing values
df_selected.dropna(inplace=True)

# Features and target
X = df_selected.drop('PCOS (Y/N)', axis=1)
y = df_selected['PCOS (Y/N)']

# Encode categorical columns if needed
if X['Blood Group'].dtype == object:
    X['Blood Group'] = LabelEncoder().fit_transform(X['Blood Group'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
with open('pcos_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as pcos_model.pkl")
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)