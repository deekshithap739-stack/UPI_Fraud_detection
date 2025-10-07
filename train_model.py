import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# 📥 Load dataset
df = pd.read_csv('sample.csv')
print("📊 Columns:", df.columns.tolist())

# 🧼 Handle missing values
df.fillna({'SenderID': 'unknown', 'ReceiverID': 'unknown', 'Amount': 0}, inplace=True)

# 🔠 Encode categorical features
le_sender = LabelEncoder()
le_receiver = LabelEncoder()
df['SenderID_encoded'] = le_sender.fit_transform(df['SenderID'])
df['ReceiverID_encoded'] = le_receiver.fit_transform(df['ReceiverID'])

# 🎯 Prepare features and target
X = df[['Amount', 'SenderID_encoded', 'ReceiverID_encoded']]
df['Fraud'] = df['Fraud'].map({'Yes': 1, 'No': 0})
y = df['Fraud']

# 🔀 Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 📊 Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# 💾 Save model and encoders
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(le_sender, 'model/le_sender.pkl')
joblib.dump(le_receiver, 'model/le_receiver.pkl')

print("✅ Model and encoders saved in model/ folder")
