import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Veriyi oku
df = pd.read_excel("HouseData2.xlsx")

# Hedef ve özellikler
y = df["fiyat"]
X = df.drop(columns=["fiyat"])

# Kategorik değişkenleri One-Hot Encode et
X_encoded = pd.get_dummies(X)
columns = X_encoded.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model eğitimi
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Modeli ve kolonları kaydet
joblib.dump(model, "model.pkl")
joblib.dump(columns, "columns.pkl")

print("Model ve kolonlar başarıyla kaydedildi.")
