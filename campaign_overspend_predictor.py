# ===============================================================
# Campaign Overspend & Underspend Predictor (Console Version)
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random

# -----------------------------
# 1️⃣ Generate sample data
# -----------------------------
data = []
for i in range(50):
    spend = random.uniform(500, 1500)
    impressions = random.randint(50000, 200000)
    clicks = random.randint(200, 1000)
    conversions = random.randint(10, 80)
    planned_budget = random.uniform(800, 1200)

    # simple logic to tag overspend / underspend / on track
    if spend > planned_budget * 1.1:
        label = "Overspend"
    elif spend < planned_budget * 0.9:
        label = "Underspend"
    else:
        label = "On Track"

    data.append([spend, impressions, clicks, conversions, planned_budget, label])

df = pd.DataFrame(data, columns=["Spend", "Impressions", "Clicks", "Conversions", "Planned_Budget", "Status"])

print("📊 Sample of your data:")
print(df.head(), "\n")

# -----------------------------
# 2️⃣ Split & train model
# -----------------------------
X = df[["Spend", "Impressions", "Clicks", "Conversions", "Planned_Budget"]]
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 3️⃣ Predict & evaluate
# -----------------------------
y_pred = model.predict(X_test)

print("✅ Model Training Complete!\n")
print("📋 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 4️⃣ Try predicting new campaigns
# -----------------------------
new_data = pd.DataFrame({
    "Spend": [1300, 600, 950],
    "Impressions": [100000, 80000, 120000],
    "Clicks": [400, 350, 500],
    "Conversions": [25, 18, 40],
    "Planned_Budget": [1000, 800, 900]
})

predictions = model.predict(new_data)
new_data["Predicted_Status"] = predictions

print("\n🔮 Predicted Spend Status for New Campaigns:")
print(new_data)
