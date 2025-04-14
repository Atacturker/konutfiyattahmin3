import streamlit as st
import pandas as pd
import joblib
import urllib.request

@st.cache_resource
def load_model():
    model_url = "https://raw.githubusercontent.com/kullaniciadi/konut-fiyat-tahmin/main/model.pkl"
    columns_url = "https://raw.githubusercontent.com/kullaniciadi/konut-fiyat-tahmin/main/columns.pkl"

    urllib.request.urlretrieve(model_url, "model.pkl")
    urllib.request.urlretrieve(columns_url, "columns.pkl")

    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

model, columns = load_model()

st.title("🏠 Konut Fiyat Tahmini")

# Kullanıcıdan veri al
ilce = st.selectbox("İlçe", ['Çankaya', 'Keçiören', 'Yenimahalle', 'Altındağ'])
mahalle = st.text_input("Mahalle")
oda_sayisi = st.selectbox("Oda Sayısı", ['1+1', '2+1', '3+1', '4+1'])
metrekare = st.number_input("Metrekare", min_value=10, max_value=1000, value=100)
bina_yasi = st.number_input("Bina Yaşı", min_value=0, max_value=100, value=10)
kat = st.number_input("Bulunduğu Kat", min_value=0, max_value=50, value=3)

if st.button("Tahmini Fiyatı Göster"):
    giris = pd.DataFrame({
        'ilce': [ilce],
        'mahalle': [mahalle],
        'oda_sayisi': [oda_sayisi],
        'metrekare': [metrekare],
        'bina_yasi': [bina_yasi],
        'kat': [kat]
    })

    giris_encoded = pd.get_dummies(giris)
    giris_encoded = giris_encoded.reindex(columns=columns, fill_value=0)

    tahmin = model.predict(giris_encoded)[0]
    st.success(f"Tahmini Fiyat: {int(tahmin):,} TL")
